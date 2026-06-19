"""Shared batch execution logic: resume, parallel workers, live dashboard."""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from . import style as S


class _Dashboard:
    """Live TTY dashboard showing running instances and summary bar."""

    def __init__(self, instance_ids: list[str]):
        self._ids = instance_ids
        self._state: dict[str, dict] = {
            iid: {"status": "waiting", "step": 0, "detail": "", "cost": 0}
            for iid in instance_ids
        }
        self._lock = threading.Lock()
        self._rendered_lines = 0

    def update(self, instance_id: str, status: str = "", step: int = 0, detail: str = "", cost: float = 0):
        with self._lock:
            s = self._state.get(instance_id, {})
            if status:
                s["status"] = status
            if step:
                s["step"] = step
            if detail:
                s["detail"] = detail
            if cost:
                s["cost"] = cost
            self._state[instance_id] = s

    def render(self):
        if not S._IS_TTY:
            return
        with self._lock:
            total_cost = 0.0
            n_done = 0
            n_failed = 0
            n_running = 0
            n_waiting = 0
            running_lines = []

            for iid in self._ids:
                s = self._state[iid]
                status = s["status"]
                cost = s.get("cost", 0)
                total_cost += cost

                if status == "done":
                    n_done += 1
                elif status == "failed":
                    n_failed += 1
                elif status == "waiting":
                    n_waiting += 1
                elif status in ("running", "spawning"):
                    n_running += 1
                    short_id = iid.split("__")[-1][:20].ljust(20)
                    step = s["step"]
                    detail = s.get("detail", "").replace("\n", " ")[:40]
                    cost_str = f"${cost:.2f}".rjust(6) if cost else "      "

                    if status == "running":
                        tag = S.neon_cyan(f"step {step:>3}")
                    else:
                        tag = S.neon_purple("spawn ")

                    running_lines.append(f"\033[K  {S.dim(short_id)} [{tag}] {S.neon_orange(cost_str)}  {detail}")

            # Summary bar
            header_parts = []
            if n_done:
                header_parts.append(S.neon_green(f"◆ {n_done} done"))
            if n_failed:
                header_parts.append(S.neon_pink(f"✘ {n_failed} failed"))
            if n_running:
                header_parts.append(S.neon_cyan(f"⚡{n_running} running"))
            if n_waiting:
                header_parts.append(S.dim(f"… {n_waiting} queued"))
            header = "  ".join(header_parts)
            total_str = S.neon_orange(f"${total_cost:.2f}")
            summary_line = f"\033[K  {header}    total: {total_str}"

            lines = [summary_line]
            if running_lines:
                lines.append(f"\033[K  {S.neon_purple('╌' * 56)}")
                lines.extend(running_lines)

            # Atomic frame write
            frame = ""
            if self._rendered_lines > 0:
                frame = f"\033[{self._rendered_lines}A"
            while len(lines) < self._rendered_lines:
                lines.append("\033[K")
            frame += "\n".join(lines) + "\n"
            sys.stdout.write(frame)
            sys.stdout.flush()
            self._rendered_lines = len(lines)

    def final_summary(self):
        """Print final non-overwritable summary."""
        if S._IS_TTY and self._rendered_lines > 0:
            sys.stdout.write(f"\033[{self._rendered_lines}A")
        total_cost = 0.0
        with self._lock:
            for iid in self._ids:
                s = self._state[iid]
                cost = s.get("cost", 0)
                total_cost += cost
                short_id = iid.split("__")[-1][:20]
                status = s["status"]
                detail = s.get("detail", "").replace("\n", " ")[:40]
                if status == "done":
                    tag = S.neon_green("◆")
                elif status == "failed":
                    tag = S.neon_pink("✘")
                else:
                    tag = S.dim("·")
                cost_str = f"${cost:.2f}" if cost else ""
                print(f"  {tag} {short_id:<22} {S.neon_orange(cost_str):>8}  {S.dim(detail)}")
            print(f"  {S.neon_purple('╌' * 56)}")
            print(f"  {'total':<24} {S.neon_orange(f'${total_cost:.2f}')}")


def run_batch(
    instances: list[dict],
    run_fn: Callable[[dict], dict],
    output_dir: Path,
    workers: int = 1,
    harness=None,
) -> list[dict]:
    """Run instances with resume support and optional parallelism.

    Args:
        instances: SWE-bench instances to run
        run_fn: function(instance) -> prediction dict
        output_dir: directory for preds.json
        workers: number of parallel workers
        harness: optional harness instance (for dashboard injection)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.json"

    # Resume: load existing predictions
    predictions = []
    if preds_path.exists():
        predictions = json.loads(preds_path.read_text())
        done_ids = {p["instance_id"] for p in predictions}
        instances = [i for i in instances if i["instance_id"] not in done_ids]
        print(f"  {S.green(S.CHECK)} Resuming: {S.bold(str(len(done_ids)))} done, "
              f"{S.bold(str(len(instances)))} remaining")

    if not instances:
        print(f"  {S.dim('nothing to do')}")
        return predictions

    def save():
        preds_path.write_text(json.dumps(predictions, indent=2))

    if workers <= 1:
        for i, inst in enumerate(instances):
            print(f"\n{S.progress(i + 1, len(instances))}")
            try:
                result = run_fn(inst)
                predictions.append(result)
                save()
            except KeyboardInterrupt:
                print(f"\n  {S.yellow('!')} Interrupted. Saving progress...")
                save()
                break
            except Exception as e:
                print(S.kv("error   ", S.bright_red(str(e))))
                predictions.append({
                    "instance_id": inst["instance_id"],
                    "model_patch": "",
                    "model_name_or_path": "error",
                    "exit_status": f"error: {e}",
                })
                save()
    else:
        # Parallel mode with live dashboard
        instance_ids = [i["instance_id"] for i in instances]
        dashboard = _Dashboard(instance_ids)

        # Inject dashboard into harness if it supports it
        if harness and hasattr(harness, "set_dashboard"):
            harness.set_dashboard(dashboard)

        # Render loop in background
        stop_render = threading.Event()

        def _render_loop():
            while not stop_render.is_set():
                dashboard.render()
                time.sleep(0.5)

        dashboard.render()
        render_thread = threading.Thread(target=_render_loop, daemon=True)
        render_thread.start()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_fn, inst): inst for inst in instances}
            try:
                for future in as_completed(futures):
                    inst = futures[future]
                    iid = inst["instance_id"]
                    try:
                        result = future.result()
                        patch = result.get("model_patch", "")
                        if patch:
                            dashboard.update(iid, status="done", detail=f"patch: {len(patch)} chars")
                        else:
                            dashboard.update(iid, status="failed", detail=result.get("exit_status", "no patch"))
                    except Exception as e:
                        result = {
                            "instance_id": iid,
                            "model_patch": "",
                            "model_name_or_path": "error",
                            "exit_status": f"error: {e}",
                        }
                        dashboard.update(iid, status="failed", detail=str(e)[:40])
                    predictions.append(result)
                    save()
            except KeyboardInterrupt:
                print(f"\n  {S.yellow('!')} Interrupted. Saving progress...")
                save()

        stop_render.set()
        render_thread.join(timeout=1)
        dashboard.final_summary()
        print()

    # Summary
    submitted = sum(1 for p in predictions if p.get("model_patch"))
    print(S.summary(len(predictions), submitted, str(preds_path)))
    return predictions
