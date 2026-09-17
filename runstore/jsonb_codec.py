"""Lossless storage for JSON strings containing PostgreSQL-forbidden NULs."""

import base64
import json

from runstore.specs import canonical


TAG = "__ash_runstore_json_utf8_base64_v1__"


def _has_nul(value):
    if isinstance(value, str):
        return "\0" in value
    if isinstance(value, dict):
        return any(_has_nul(key) or _has_nul(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_nul(item) for item in value)
    return False


def dumps(value):
    raw = canonical(value)
    # Escape literal envelope-shaped values too, so decoding remains bijective.
    if _has_nul(value) or isinstance(value, dict) and set(value) == {TAG}:
        return canonical({TAG: base64.b64encode(raw.encode("utf-8")).decode("ascii")})
    return raw


def loads(text):
    value = json.loads(text)
    if isinstance(value, dict) and set(value) == {TAG}:
        # Decode only the outer storage envelope. A literal nested envelope is
        # user data and must remain untouched.
        return json.loads(base64.b64decode(value[TAG], validate=True).decode("utf-8"))
    return value


def diagnostic_text(value):
    return value.replace("\0", "\\u0000") if isinstance(value, str) else value
