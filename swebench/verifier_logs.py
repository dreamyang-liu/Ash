"""Retain complete verifier streams without changing command exit status."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import tempfile
from uuid import uuid4


class VerifierLogSession:
    def __init__(self, session, root: Path, guest_root: str = "/tmp"):
        self.session = session
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="verify-", dir=root)).resolve()
        self.guest = guest_root + "/ash-swe-verifier-" + uuid4().hex
        self.calls = 0
        self.errors = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        prepared = session.execute("shell", {
            "command": "command -v bash && command -v tee && command -v tar && mkdir -p " + shlex.quote(self.guest),
            "timeout": 30,
        })
        self.capture = prepared.success
        if not self.capture:
            self.errors.append("log directory preparation failed: " + str(prepared.error or prepared.output))

    def execute(self, tool_name: str, args: dict, **kwargs):
        self.calls += 1
        number = self.calls
        actual = dict(args)
        if tool_name == "shell" and self.capture:
            stdout = shlex.quote(f"{self.guest}/{number:04}.stdout")
            stderr = shlex.quote(f"{self.guest}/{number:04}.stderr")
            script = ("sh -c " + shlex.quote(str(args["command"]))
                      + f" > >(tee {stdout}) 2> >(tee {stderr} >&2); "
                      + "command_status=$?; wait; exit \"$command_status\"")
            actual["command"] = "bash -c " + shlex.quote(script)
        record = {"number": number, "tool": tool_name, "args": args,
                  "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            result = self.session.execute(tool_name, actual, **kwargs)
            record.update(success=result.success, output=result.output, error=result.error)
            return result
        except BaseException as error:
            record["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            try:
                with (self.directory / "calls.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            except OSError as error:
                self.errors.append(str(error))

    def finish(self, grade) -> None:
        archive = self.guest + ".tar.gz"
        grade.verifier_artifacts = str(self.directory)
        try:
            if self.capture:
                result = self.session.execute("shell", {
                    "command": "tar -czf " + shlex.quote(archive) + " -C " + shlex.quote(self.guest) + " .",
                    "timeout": 120,
                })
                if not result.success:
                    self.errors.append("archive failed: " + str(result.error or result.output))
                elif not self.session.download_file(archive, self.directory / "verifier-logs.tar.gz"):
                    self.errors.append("archive download failed")
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
        grade.verifier_artifact_error = "; ".join(self.errors) or None
        metadata = {"started_at": self.started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
                    "sandbox_id": getattr(self.session, "sandbox_id", None), "calls": self.calls,
                    "resolved": grade.resolved, "grading_error": grade.error,
                    "artifact_error": grade.verifier_artifact_error}
        try:
            (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        except OSError as error:
            self.errors.append(str(error))
            grade.verifier_artifact_error = "; ".join(self.errors)
