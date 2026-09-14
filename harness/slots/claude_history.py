"""Immutable native transcript prefixes registered as independent SDK sessions."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4


@dataclass(frozen=True)
class PrefixSource:
    transcript: Path
    sha256: str
    session_id: str
    cut: str
    referenced_outputs: tuple[tuple[str, str], ...] = ()


def _read_prefix(transcript: Path, cut: str) -> tuple[bytes, list[dict]]:
    data = transcript.read_bytes()
    entries = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError("native transcript entry is not an object")
        entries.append(entry)
        if entry.get("uuid") == cut:
            return data, entries
    raise ValueError("native prefix cut is missing")


def _referenced_outputs(entries: list[dict], transcript: Path, allowed_outputs=()) -> list[dict]:
    references = []
    decoded_text = "\n".join(_strings(entries))
    for path_text in sorted(set(re.findall(r"Full output saved to: ([^\n\r]+)", decoded_text))):
        path = Path(path_text.strip())
        pinned = dict(allowed_outputs)
        if not path.is_absolute() or (not path.resolve().is_relative_to(transcript.parent.parent.resolve())
                                      and str(path) not in pinned):
            raise ValueError("unsupported persisted-output reference in native prefix")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in pinned and checksum != pinned[str(path)]:
            raise ValueError("referenced native output changed after branch selection")
        references.append({"path": str(path), "sha256": checksum})
    return references


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def find_prefix_source(projects: Path, session_id: str, cut: str) -> PrefixSource | None:
    for transcript in projects.glob("*/%s.jsonl" % session_id):
        try:
            data, entries = _read_prefix(transcript, cut)
            references = _referenced_outputs(entries, transcript)
        except (OSError, ValueError):
            continue
        return PrefixSource(transcript, hashlib.sha256(data).hexdigest(), session_id, cut,
                            tuple((item["path"], item["sha256"]) for item in references))
    return None


def prepare_prefix(source: PrefixSource, cwd: Path, receipt_dir: Path, projects: Path) -> dict:
    """Register only the selected history; never rewrite the source session."""
    from claude_agent_sdk._internal.sessions import _sanitize_path

    data, entries = _read_prefix(source.transcript, source.cut)
    if hashlib.sha256(data).hexdigest() != source.sha256:
        raise ValueError("native transcript changed after branch selection")
    references = _referenced_outputs(entries, source.transcript, source.referenced_outputs)
    if tuple((item["path"], item["sha256"]) for item in references) != source.referenced_outputs:
        raise ValueError("referenced native output changed after branch selection")
    cwd = cwd.resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=False)
    session_id = str(uuid4())
    project = projects / _sanitize_path(str(cwd))
    project.mkdir(parents=True, exist_ok=True)
    native_path = project / (session_id + ".jsonl")
    copied = []
    for original in entries:
        entry = {**original, "sessionId": session_id}
        if "cwd" in entry:
            entry["cwd"] = str(cwd)
        copied.append(entry)
    text = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in copied) + "\n"
    saved_prefix = receipt_dir / "prefix.jsonl"
    saved_prefix.write_text(text, encoding="utf-8")
    created = False
    try:
        with native_path.open("x", encoding="utf-8") as stream:
            created = True
            stream.write(text)
    except BaseException:
        if created:
            native_path.unlink(missing_ok=True)
        raise
    manifest = {
        "mode": "original-prefix", "source_transcript": str(source.transcript),
        "source_sha256": source.sha256, "source_session_id": source.session_id,
        "cut": source.cut, "resume_session_id": session_id, "cwd": str(cwd),
        "native_path": str(native_path), "saved_prefix": str(saved_prefix),
        "prefix_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "entry_count": len(copied), "referenced_outputs": references,
    }
    manifest_path = receipt_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}
