#!/usr/bin/env python3
"""Generate docs/architecture-overview.excalidraw — architecture + sequence.

One canvas, two halves:

- top: the full layered architecture (reused verbatim from gen_architecture.py's
  build -- two pictures of the architecture is how they diverge);
- bottom: a sequence diagram (泳道时序图) of one run and its fork, exactly the
  flow verified live on SWE-bench Verified.

    python3 docs/gen_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_architecture  # noqa: E402
from exgen import (BLUE, GREEN, GREY, ORANGE, RED, VIOLET, WHITE,  # noqa: E402
                   Canvas, text_size)

OUT = Path(__file__).resolve().parent / "architecture-overview.excalidraw"


class Sequence:
    """A swim-lane sequence diagram on a Canvas.

    Actors are columns; time flows down one row per message. Rows are allocated
    by a counter, so inserting a message renumbers everything below it -- no
    hand-maintained y coordinates, which is the class of mistake the validator
    exists to catch after the fact rather than prevent.
    """

    ROW = 34

    def __init__(self, c: Canvas, x: float, y: float, actors: list,
                 col_w: float = 208):
        self.c = c
        self.x = x
        self.col_w = col_w
        self.actors = [a[0] for a in actors]
        self._x = {}
        self._row = 0
        header_bottom = y
        for i, (key, title, color) in enumerate(actors):
            cx = x + i * col_w
            p = c.panel("seq-h-%s" % key, cx, y, col_w - 18, title, "",
                        color=color, title_size=12)
            self._x[key] = p["cx"]
            header_bottom = max(header_bottom, p["bottom"])
        self.top = header_bottom + 6
        self._headers_bottom = header_bottom

    def _y(self, row: int) -> float:
        return self.top + row * self.ROW

    def message(self, src: str, dst: str, label: str, *, color: str = WHITE,
                dashed: bool = False, size: int = 11) -> None:
        self._row += 1
        y = self._y(self._row)
        x0, x1 = self._x[src], self._x[dst]
        if src == dst:
            # Self-message: a zero-length arrow is invisible, so draw a small
            # out-and-back loop to the right of the lifeline.
            self.c.arrow("seq-m%d" % self._row, x0, y - 8,
                         [[0, 0], [26, 0], [26, 12], [0, 12]],
                         color=color, dashed=dashed)
            self.c.text("seq-l%d" % self._row, x0 + 34, y - 12, label,
                        size=size, color=color)
            return
        self.c.arrow("seq-m%d" % self._row, x0, y, [[0, 0], [x1 - x0, 0]],
                     color=color, dashed=dashed)
        width, _ = text_size(label, size)
        left = min(x0, x1)
        span = abs(x1 - x0)
        # centre the label on its arrow; a label wider than the span still
        # starts inside it, which the free-text collision check then polices.
        lx = left + max(6, (span - width) / 2)
        self.c.text("seq-l%d" % self._row, lx, y - 16, label, size=size,
                    color=GREY if dashed else WHITE)

    def note(self, label: str, *, color: str = GREY) -> None:
        self._row += 1
        y = self._y(self._row) - 12
        self.c.text("seq-n%d" % self._row, self.x + 6, y, label, size=11,
                    color=color)

    def frame_start(self) -> float:
        """Open a loop/section frame; reserves one row for the frame's label so
        it cannot collide with the first message inside."""
        y0 = self._y(self._row) + 10
        self._row += 1
        return y0

    def frame(self, id_: str, y0: float, label: str, *, color: str = VIOLET) -> None:
        """Close a frame opened at ``y0`` around everything drawn since."""
        y1 = self._y(self._row) + 14
        x0 = self.x - 8
        x1 = self.x + len(self.actors) * self.col_w - 10
        self.c.box(id_, x0, y0, x1 - x0, y1 - y0, color=color, dashed=True)
        # Inside the frame, on the reserved row -- outside it collides with the
        # label of whatever message precedes the frame.
        self.c.text(id_ + "-t", x0 + 12, y0 + 6, label, size=12, color=color)

    def lifelines(self) -> float:
        """Draw the dashed lifelines last, from headers to the final row."""
        bottom = self._y(self._row) + 26
        for key in self._x:
            self.c.arrow("seq-life-%s" % key, self._x[key],
                         self._headers_bottom + 2,
                         [[0, 0], [0, bottom - self._headers_bottom - 4]],
                         color="#555555", dashed=True, head=None)
        return bottom


def main() -> int:
    # ---- top half: the architecture, verbatim -------------------------------
    gen_architecture.build()
    c = gen_architecture._c
    arch_bottom = max(e["y"] + e["height"] for e in c.els)

    # ---- bottom half: the sequence ------------------------------------------
    sy = arch_bottom + 80
    c.text("seq-title", 40, sy,
           "时序:一次 run + 一次 fork(SWE-bench Verified 真机验证过的流程)",
           size=20, color=BLUE)
    c.text("seq-sub", 40, sy + 32,
           "实线 = 调用/数据 · 虚线 = 事件流 · 紫框 = 循环 · 时序图与上面架构图的对应:列即组件",
           size=12, color=GREY)

    seq = Sequence(c, 40, sy + 66, [
        ("caller", "调用方\nCLI/demo_fork", BLUE),
        ("orch", "Orchestrator", GREEN),
        ("agent", "Agent slot\nclaude|codex|oc", BLUE),
        ("mcp", "MCP Server\n进程内|子进程", GREEN),
        ("vm", "microVM\n(AgentENV)", ORANGE),
        ("ckpt", "Checkpointer\ntracker+bridge", ORANGE),
        ("journal", "Journal\nJSONL 真源", ORANGE),
    ])

    seq.message("caller", "orch", "RunSpec{image, transport, tools}")
    seq.message("orch", "journal", "开 journal(分支则先写 fork.origin)")
    seq.message("orch", "vm", "session.create(image) —— 模板按需构建,VM 起来")
    seq.message("orch", "mcp", "起 server · adopt(handle) / --attach id · 面板+tracker 挂上")
    seq.message("orch", "ckpt", "SnapshotBridge.install(同一个 tracker)")
    seq.message("orch", "agent", "slot.run(prompt, MCP wiring)")

    loop_y = seq.frame_start()
    seq.message("agent", "mcp", "tools/call {name, args}")
    seq.message("mcp", "vm", "route → 洋葱(tracker 记 dirty)→ 执行")
    seq.message("vm", "mcp", "ToolResult")
    seq.message("mcp", "agent", "content(isError 在 result 层,不在块里)")
    seq.message("mcp", "ckpt", "ToolBoundary.after_step(n)")
    seq.message("ckpt", "vm", "dirty→snapshot(disk_only);clean→复用上一张")
    seq.message("ckpt", "journal", "checkpoint.captured{step,snap,会话ref}")
    seq.frame("seq-loop1", loop_y, "循环:每个 exec 工具调用(改动步拍照,只读步复用)")

    seq.message("agent", "journal", "事件流(normalize:message/turn/usage)",
                dashed=True)
    seq.message("agent", "orch", "SlotResult")
    seq.message("orch", "mcp", "stop(server 随 run 死)")
    seq.message("orch", "vm", "destroy(沙箱亡,快照存)")
    seq.message("orch", "caller", "RunOutcome{journal 路径, pairs 数}")

    fork_y = seq.frame_start()
    seq.message("caller", "journal", "fork_plan(step N) → {snapshot, 会话 ref}")
    seq.message("caller", "orch",
                "RunSpec{image=snapshot, resume+fork} —— 回到第 2 行,整段重演")
    seq.note("   每个分支:新 microVM 从快照起(环境半边)+ 会话 fork(对话半边)"
             " → 兄弟分支两个半边都隔离", color=VIOLET)
    seq.frame("seq-loop2", fork_y, "循环:每个分支方向(× K)")

    seq.note("真机验证(django__django-11848):第 1 步快照恢复出干净树,第 2 步恢复出 fix;"
             "分岔出「验证原修法(45 测试过)」和「推翻重写」两个最终 diff 各异的分支",
             color=GREEN)
    seq.lifelines()

    c.write(OUT, containers={"l2", "l3", "seq-loop1", "seq-loop2"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
