import json

import pytest

from runstore.jsonb_codec import TAG, dumps, loads
from runstore.specs import canonical, digest


@pytest.mark.parametrize("value", [
    {"text": "\x7fELF\x02\x01\x01\0binary\0tail"},
    {"nested": [{"value": "a\0b"}, {"ordinary": "中文"}]},
    {"key\0withnul": "value"},
    {TAG: "a literal envelope-shaped user value"},
    {"text": "\\u0000 is literal, \0 is not"},
])
def test_lossless_roundtrip_and_hash_stability(value):
    stored = dumps(value)
    assert loads(stored) == value
    assert digest(loads(stored)) == digest(value)
    assert "\0" not in json.dumps(json.loads(stored), ensure_ascii=False)


def test_ordinary_json_is_unchanged():
    value = {"type": "tool.finished", "output": "ordinary result"}
    assert dumps(value) == canonical(value)
    assert loads(canonical(value)) == value
