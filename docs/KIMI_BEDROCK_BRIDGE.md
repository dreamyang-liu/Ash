# Kimi K3 native chat bridge

`scripts/kimi_bedrock_bridge.py` serves the chat interface consumed by the mini
slot. It sends native chat payloads to Bedrock `InvokeModel`, retaining ordinary
assistant content, `reasoning_content`, tool calls and tool-result IDs without
converting the history through Converse.

The model is pinned to `us.moonshotai.kimi-k3`, reasoning effort to `max`, and
output requests are bounded at 64,000 tokens. Only non-streaming requests are
accepted. Tool-name/schema validation remains the responsibility of the mini
runtime and branching controller.

## Run

Install the bridge's optional dependencies into its Python environment:

```bash
pip install boto3 fastapi uvicorn
python scripts/kimi_bedrock_bridge.py \
  --token-file /path/to/bridge-token \
  --records /path/to/private/bridge-records \
  --profile my-aws-profile \
  --region us-east-1 \
  --port 18239
```

The token file contains the bearer token used by callers. Keep tokens, AWS
credentials and recorded conversations outside the repository. The script uses
the standard SDK credential chain; `--profile` is optional and standard AWS
environment variables are honored. It contains no machine-specific credential
paths.

`GET /health` reports the model and timeout settings. Mini clients use
`POST /v1/chat/completions` with their bearer token and the pinned model name.

## Failure handling

The SDK read timeout defaults to 360 seconds and is configurable through
`--read-timeout`. Automatic SDK retries are disabled. Changing this timeout for
an evaluation changes its transport policy and should be recorded separately.

The exception handler accepts absent, null or malformed SDK error metadata:

- Read/connect timeouts return HTTP 504.
- Throttling returns HTTP 429.
- Other valid upstream error statuses are retained; unclassified errors return 502.

Every invocation gets a request ID and saved request. Successful replies and
structured error events include elapsed time. A timeout is recorded even when
the exception's `response` attribute is explicitly `None`; this fixes the
secondary handler crash that previously returned an unrecorded HTTP 500.

This correction improves failure classification and accounting. It does not
make slow requests finish sooner, retry interrupted generations, resume a
paused batch or reset any actor budget.

## Verify

```bash
PYTHONPATH=.:sdk python -m pytest harness/tests/test_kimi_bedrock_bridge.py -q
```

Tests inject a fake SDK client and make no AWS calls. They cover native history
preservation, error classification/logging, no retries, request rejection and
startup argument validation. The tests skip when optional bridge dependencies
are not installed.
