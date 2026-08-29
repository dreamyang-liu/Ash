#!/usr/bin/env python3
"""A field-stripping proxy for OpenAI Responses API traffic.

Why this exists: codex's Responses requests carry codex-only fields
(``client_metadata``), and translation layers (litellm 1.97/1.98, ``--drop_params``
included) forward them verbatim into providers that reject unknown fields --
Bedrock's Anthropic backend answers "client_metadata: Extra inputs are not
permitted" and the run dies. Until a translator handles it, this sits in front:
strip the named top-level fields, forward everything else byte-identical,
relay streaming responses as they arrive.

    python3 scripts/responses_scrub.py --port 4478 \
        --upstream http://127.0.0.1:4477 --strip client_metadata

Deliberately dumb: no auth (it forwards yours), no routing, no accounting --
that is the inference gateway's job. This is a shim for one wire-format
incompatibility, and should be deleted the day the translator handles the field.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4478)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--strip", action="append", default=None,
                        metavar="FIELD",
                        help="top-level JSON fields to drop (repeatable; "
                             "default: client_metadata)")
    args = parser.parse_args()
    strip = set(args.strip or ["client_metadata"])
    upstream = args.upstream.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: A003
            pass

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                payload = json.loads(body)
                dropped = [k for k in strip if payload.pop(k, None) is not None]
                if dropped:
                    body = json.dumps(payload).encode()
            except ValueError:
                pass          # not JSON: forward untouched
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length")}
            headers["Content-Length"] = str(len(body))
            request = urllib.request.Request(upstream + self.path, data=body,
                                             headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=600)
            except urllib.error.HTTPError as err:
                response = err
            self.send_response(response.status)
            hop = {"transfer-encoding", "connection", "content-length"}
            for key, value in response.headers.items():
                if key.lower() not in hop:
                    self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

        def do_GET(self):  # noqa: N802
            request = urllib.request.Request(
                upstream + self.path,
                headers={k: v for k, v in self.headers.items()
                         if k.lower() != "host"})
            try:
                response = urllib.request.urlopen(request, timeout=60)
            except urllib.error.HTTPError as err:
                response = err
            content = response.read()
            self.send_response(response.status)
            self.send_header("Content-Length", str(len(content)))
            for key, value in response.headers.items():
                if key.lower() not in ("transfer-encoding", "connection",
                                       "content-length"):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write("[scrub] %s:%d -> %s (strip: %s)\n"
                     % (args.host, args.port, upstream, ", ".join(sorted(strip))))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
