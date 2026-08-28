#!/usr/bin/env python3
"""Generate docs/checkpoint-flow.excalidraw — one tool call, start to journal.

The flow diagram for the unified checkpoint mechanism: a single vertical spine
(the machinery both transports share) that forks exactly twice -- at the top for
*which process serves the call*, at the bottom for *where the record lands*.
That shape IS the design claim: everything between the forks is one code path.

    python3 docs/gen_checkpoint_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exgen import (BLUE, GREEN, GREY, ORANGE, RED, VIOLET, WHITE,  # noqa: E402
                   Canvas, text_size)

OUT = Path(__file__).resolve().parent / "checkpoint-flow.excalidraw"

GAP = 26          # vertical gap between spine steps (room for one arrow)
COL = 24


def down_arrow(c: Canvas, id_: str, x: float, y_from: float, y_to: float,
               label: str = "", *, color: str = WHITE,
               dashed: bool = False) -> None:
    c.arrow(id_, x, y_from + 4, [[0, 0], [0, y_to - y_from - 8]],
            color=color, dashed=dashed)
    if label:
        c.text(id_ + "-l", x + 10, y_from + (y_to - y_from) / 2 - 8, label,
               size=11, color=GREY)


def main() -> int:
    c = Canvas()

    c.text("title", 40, 24, "一次工具调用的流程（checkpoint 机制，两条 transport）",
           size=20, color=BLUE)
    legend = ("一条中轴 = 两条 transport 共用的机制（同一份代码）。只在两处分岔：\n"
              "  ① 顶部 —— 这次调用由哪个进程服务    ② 底部 —— checkpoint 记录落在哪\n"
              "紫 = transport 专属 · 绿 = 共用机制 · 橙 = 状态/记录 · 红 = 曾经错过的地方")
    c.text("legend", 40, 58, legend, size=12, color=GREY)
    y = 58 + text_size(legend, 12)[1] + 30

    # 中轴的横向位置：所有主干框共享同一个中线
    SPINE_X = 330
    SPINE_W = 560
    spine_cx = SPINE_X + SPINE_W / 2

    # ---- agent ---------------------------------------------------------------
    agent = c.panel("agent", SPINE_X, y, SPINE_W,
                    "Agent（claude-code / codex / opencode）",
                    "MCP tools/call {name, arguments} —— 它跟环境的唯一通道，"
                    "所以一次工具调用 = 它的一步",
                    color=BLUE)

    # ---- fork #1: which process serves ---------------------------------------
    fy = agent["bottom"] + 44
    http_p = c.panel("http", 40, fy, 400, "分岔① http：本进程服务",
                     "orchestrator 进程内的 HttpMcpServer\n"
                     "沙箱用 handle 交出（pool.adopt）\n"
                     "机制由 orchestrator 构造（_serve_in_process）",
                     color=VIOLET)
    stdio_p = c.panel("stdio", http_p["right"] + 200, fy, 400,
                      "分岔① stdio：子进程服务",
                      "slot 拉起的 server 子进程（严格串行 →\n"
                      "  钩子响的时候必然没有调用在飞）\n"
                      "沙箱按 id 交出（--attach，microvm）\n"
                      "机制由 server main 构造（--checkpoint-log）",
                      color=VIOLET)
    # agent -> both
    c.arrow("a-h", agent["cx"] - 60, agent["bottom"] + 4,
            [[0, 0], [http_p["cx"] - agent["cx"] + 60, fy - agent["bottom"] - 8]],
            color=VIOLET)
    c.arrow("a-s", agent["cx"] + 60, agent["bottom"] + 4,
            [[0, 0], [stdio_p["cx"] - agent["cx"] - 60, fy - agent["bottom"] - 8]],
            color=VIOLET)

    # ---- the shared spine ------------------------------------------------------
    sy = max(http_p["bottom"], stdio_p["bottom"]) + 44
    c.text("shared-t", SPINE_X, sy - 24,
           "———— 从这里到分岔② 之前：同一份代码，两个进程各跑一份 ————",
           size=13, color=GREEN)

    route = c.panel("route", SPINE_X, sy, SPINE_W,
                    "1  ExecSurface.route()",
                    "改名视图还原成 runtime 工具名（run_tests → shell），\n"
                    "先于一切 —— 后面的 tracker、钩子、沙箱才认得这个名字",
                    color=GREEN)
    # converge arrows
    c.arrow("h-r", http_p["cx"], http_p["bottom"] + 4,
            [[0, 0], [route["cx"] - http_p["cx"], sy - http_p["bottom"] - 8]],
            color=VIOLET)
    c.arrow("s-r", stdio_p["cx"], stdio_p["bottom"] + 4,
            [[0, 0], [route["cx"] - stdio_p["cx"], sy - stdio_p["bottom"] - 8]],
            color=VIOLET)

    py = route["bottom"] + GAP
    pipe = c.panel("pipe", SPINE_X, py, SPINE_W,
                   "2  pipeline（洋葱）—— MutationTracker.before",
                   "interceptor 记下「这步可能改了环境」：\n"
                   "  shell 一律算改（ls 和 ls > out 从文本上分不出来）\n"
                   "  text_editor view / grep / web_* 不算改\n"
                   "  shell(background) 记「可能有后台进程」→ 落进记录，replay 工具会警告",
                   color=GREEN)
    down_arrow(c, "r-p", spine_cx, route["bottom"], py)

    ey = pipe["bottom"] + GAP
    execu = c.panel("exec", SPINE_X, ey, SPINE_W,
                    "3  沙箱执行，结果回给 agent",
                    "executor → ash-runtime（microVM 里）→ ToolResult",
                    color=GREEN)
    down_arrow(c, "p-e", spine_cx, pipe["bottom"], ey)

    by = execu["bottom"] + GAP
    bound = c.panel("bound", SPINE_X, by, SPINE_W,
                    "4  ToolBoundary.after_call() —— 每个 exec 调用后，不只「改动型」",
                    "step += 1，worker 线程跑同步的 Checkpointer（loop 套不进 loop）\n"
                    "失败只记日志：checkpoint 是给事后分析的优化，\n"
                    "  绝不能把 agent 的工具调用搞挂\n"
                    "只读的步也要触发 —— 它也需要 map 条目，不然没法在那步 fork",
                    color=GREEN)
    down_arrow(c, "e-b", spine_cx, execu["bottom"], by)

    # ---- the decision -----------------------------------------------------------
    dy = bound["bottom"] + GAP
    decide = c.panel("decide", SPINE_X, dy, SPINE_W,
                     "5  Checkpointer.after_step(step) —— tracker.dirty ?",
                     "问的是那个 interceptor，不是猜的 —— 这就是「一套机制」的接缝",
                     color=GREEN)
    down_arrow(c, "b-d", spine_cx, bound["bottom"], dy)

    ry = decide["bottom"] + 40
    clean = c.panel("clean", 40, ry, 400, "没改（clean）",
                    "不拍。map[step] = 上一张快照的 id\n"
                    "记录照写（captured=false, reason=clean）\n"
                    "→ map 完整，但不为读操作花钱",
                    color=ORANGE)
    capture = c.panel("capture", clean["right"] + 200, ry, 400, "改了（dirty）",
                      "session.snapshot(disk_only) → 新快照\n"
                      "tracker.clear()\n"
                      "层链被服务端压缩了 → re-board 换沙箱\n"
                      "层数 ≥128 → squash 折叠链",
                      color=ORANGE)
    c.arrow("d-cl", decide["cx"] - 60, decide["bottom"] + 4,
            [[0, 0], [clean["cx"] - decide["cx"] + 60, ry - decide["bottom"] - 8]])
    c.text("d-cl-l", clean["cx"] + 40, decide["bottom"] + 8, "否", size=12,
           color=GREY)
    c.arrow("d-cp", decide["cx"] + 60, decide["bottom"] + 4,
            [[0, 0], [capture["cx"] - decide["cx"] - 60, ry - decide["bottom"] - 8]])
    c.text("d-cp-l", capture["cx"] - 60, decide["bottom"] + 8, "是", size=12,
           color=GREY)

    # ---- both outcomes produce the same record --------------------------------
    rec_y = max(clean["bottom"], capture["bottom"]) + GAP
    record = c.panel("record", SPINE_X, rec_y, SPINE_W,
                     "6  一条记录：{step, snapshot_id, captured, reason}",
                     "clean 和 capture 产出同一种记录 —— 差别只在 captured 标志和\n"
                     "指向哪张快照。每一步都有一条，map 才是完整的",
                     color=GREEN)
    c.arrow("cl-rec", clean["cx"], clean["bottom"] + 4,
            [[0, 0], [record["cx"] - clean["cx"], rec_y - clean["bottom"] - 8]])
    c.arrow("cp-rec", capture["cx"], capture["bottom"] + 4,
            [[0, 0], [record["cx"] - capture["cx"], rec_y - capture["bottom"] - 8]])

    # ---- fork #2: where the record lands ------------------------------------------
    ky = record["bottom"] + 44
    c.text("sink-t", SPINE_X, ky - 24,
           "———— 分岔②：记录落在哪 ————", size=13, color=VIOLET)
    sink_h = c.panel("sink-h", 40, ky, 400, "分岔② http：直接进 journal",
                     "bridge._on_checkpoint → RollbackLedger\n"
                     "→ journal 事件 checkpoint.captured\n"
                     "（同进程，journal 就在手边）",
                     color=VIOLET)
    sink_s = c.panel("sink-s", sink_h["right"] + 200, ky,
                     400, "分岔② stdio：JSONL → 父进程实时 tail",
                     "子进程手边没有 journal → 每步追加一行\n"
                     "  到 --checkpoint-log（当场写，被 kill 保住已写的）\n"
                     "父进程 CheckpointTail 边跑边读：行一出现就折进\n"
                     "  journal（实测 run 进行到 17s 时 pair 已在），\n"
                     "  会话 ref 晚到就走 bridge 的回填",
                     color=VIOLET)
    c.arrow("rec-sh", record["cx"] - 60, record["bottom"] + 4,
            [[0, 0], [sink_h["cx"] - record["cx"] + 60, ky - record["bottom"] - 8]],
            color=VIOLET)
    c.arrow("rec-ss", record["cx"] + 60, record["bottom"] + 4,
            [[0, 0], [sink_s["cx"] - record["cx"] - 60, ky - record["bottom"] - 8]],
            color=VIOLET)

    # ---- convergence -----------------------------------------------------------------
    jy = max(sink_h["bottom"], sink_s["bottom"]) + 44
    journal = c.panel("journal", SPINE_X, jy, SPINE_W,
                      "journal：checkpoint.captured {step, snapshot_id, session_ckpt}",
                      "环境快照 + 会话 ref = 一个完整的 rollback pair\n"
                      "load_checkpoints / fork_plan 对两种 run 的读法完全一致 ——\n"
                      "在任意一步 restore、branch、事后提取",
                      color=ORANGE)
    c.arrow("sh-j", sink_h["cx"], sink_h["bottom"] + 4,
            [[0, 0], [journal["cx"] - sink_h["cx"], jy - sink_h["bottom"] - 8]])
    c.arrow("ss-j", sink_s["cx"], sink_s["bottom"] + 4,
            [[0, 0], [journal["cx"] - sink_s["cx"], jy - sink_s["bottom"] - 8]])

    # ---- the lesson -------------------------------------------------------------------
    ly = journal["bottom"] + 30
    c.panel("lesson", 40, ly, 1060,
            "红线：这张图之前不是这个形状，两边各错一半（都是静默的）",
            "曾经是两套散装 mixin，都挂在 pool 钩子上、都绕开了 tracker + Checkpointer：\n"
            "  http 侧 tracker 建了没挂上 pipeline、没人喂 → 不开 always 时每步都判 clean，"
            "agent 在写文件而 map 说环境没变\n"
            "  stdio 侧手写捕获 → view 白拍一张、grep 那步没有 map 条目、层链维护全没有\n"
            "真机验证（不开 always）：write→拍 · view→clean 复用 · write→再拍，两条 transport 逐条一致",
            color=RED)

    c.write(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
