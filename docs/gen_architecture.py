#!/usr/bin/env python3
"""Generate docs/architecture-current.excalidraw.

Checked in because the previous diagram was hand-written JSON, which went wrong in
two ways worth avoiding: typos that made the file unopenable, and -- worse -- text
that silently overflowed its box or boxes that overlapped, so the picture looked
fine to whoever wrote it and was unreadable to whoever opened it.

Everything here is laid out from a declared grid, and :func:`validate` refuses to
write a file whose boxes overlap, whose text escapes its box, or whose labels sit
on top of each other. Run it after changing the architecture:

    python3 docs/gen_architecture.py

Font metrics are approximations of Excalidraw's default (Virgil/Excalifont) at the
sizes used here, deliberately pessimistic: a CJK glyph is a full em, latin ~0.58,
so the checker over-estimates width rather than passing something that will clip.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parent / "architecture-current.excalidraw"

# --- palette ---------------------------------------------------------------
BLUE = "#4dabf7"      # callers / titles
GREEN = "#51cf66"     # execution plane
ORANGE = "#ffa94d"    # state
YELLOW = "#ffd43b"    # benchmark boundary
GREY = "#a0a0a0"      # prose
WHITE = "#e8e8e8"     # body text, solid arrows
VIOLET = "#b197fc"    # the two transports
RED = "#ff8787"        # the asymmetry / warnings

_seed = [0]


def _next_seed() -> int:
    _seed[0] += 1
    return _seed[0]


def _char_w(ch: str, size: int) -> float:
    """Approximate advance width of one character."""
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return size * 1.0
    return size * 0.58


def text_size(body: str, size: int) -> "tuple[float, float]":
    lines = body.split("\n")
    width = max((sum(_char_w(c, size) for c in line) for line in lines), default=0)
    return width, len(lines) * size * 1.25


def text(id_: str, x: float, y: float, body: str, *, size: int = 13,
         color: str = WHITE, group: "str | None" = None) -> dict:
    width, height = text_size(body, size)
    return {
        "id": id_, "type": "text", "x": x, "y": y,
        "width": round(width, 1), "height": round(height, 1),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [group] if group else [],
        "roundness": None, "seed": _next_seed(), "version": 1,
        "isDeleted": False, "text": body, "fontSize": size, "fontFamily": 1,
        "textAlign": "left", "verticalAlign": "top", "containerId": None,
        "originalText": body, "lineHeight": 1.25, "boundElements": [],
        "link": None, "locked": False, "autoResize": True,
    }


def box(id_: str, x: float, y: float, w: float, h: float, *, color: str = BLUE,
        dashed: bool = False, group: "str | None" = None) -> dict:
    return {
        "id": id_, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1, "opacity": 100, "groupIds": [group] if group else [],
        "roundness": {"type": 3}, "seed": _next_seed(), "version": 1,
        "isDeleted": False, "boundElements": [], "link": None, "locked": False,
    }


def arrow(id_: str, x: float, y: float, points: list, *, color: str = WHITE,
          dashed: bool = False) -> dict:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "id": id_, "type": "arrow", "x": x, "y": y,
        "width": max(xs) - min(xs), "height": max(1, max(ys) - min(ys)),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "roundness": {"type": 2},
        "seed": _next_seed(), "version": 1, "isDeleted": False,
        "points": points, "lastCommittedPoint": None, "startBinding": None,
        "endBinding": None, "startArrowhead": None, "endArrowhead": "arrow",
        "boundElements": [], "link": None, "locked": False,
    }


# --- the diagram -----------------------------------------------------------
PAD = 14          # inner padding between a box edge and its text
els: list = []
# (box_id, [text_id, ...]) -- validate() checks each text fits its box
CONTAINS: list = []


def panel(id_: str, x: float, y: float, w: float, title: str, body: str, *,
          color: str = BLUE, dashed: bool = False, title_size: int = 14,
          body_size: int = 12) -> dict:
    """A titled box sized to its contents. Returns its geometry.

    ``w`` is a *minimum*: the box widens to whatever its text needs. Trusting a
    hand-picked width is how the previous diagram ended up with clipped lines --
    the width looked plausible, and nothing checked.
    """
    tw, th = text_size(title, title_size)
    bw, bh = text_size(body, body_size) if body else (0, 0)
    w = max(w, tw + 2 * PAD + 6, bw + 2 * PAD + 6)
    h = PAD + th + (6 + bh if body else 0) + PAD
    els.append(box(id_, x, y, w, h, color=color, dashed=dashed))
    ids = []
    els.append(text(id_ + "-t", x + PAD, y + PAD, title, size=title_size,
                    color=color))
    ids.append(id_ + "-t")
    if body:
        els.append(text(id_ + "-b", x + PAD, y + PAD + th + 6, body,
                        size=body_size))
        ids.append(id_ + "-b")
    CONTAINS.append((id_, ids))
    return {"id": id_, "x": x, "y": y, "w": w, "h": h,
            "right": x + w, "bottom": y + h}


def build() -> None:
    """Lay out top to bottom, each row from the previous row's measured bottom.

    Nothing here hard-codes a coordinate that depends on how tall some other box
    turned out to be. The previous diagram did, which is why a one-line edit to a
    label could push a box out of its container with no visible sign.
    """
    GAP = 16
    COL = 20          # gap between columns
    els.append(text("title", 40, 24,
                    "Ash 架构（feat/harness-v1，本轮改动后）", size=20, color=BLUE))
    legend = ("绿 = 执行平面 · 蓝 = 调用方 · 橙 = 状态 · 黄 = benchmark 插件 · "
              "紫 = 两条 transport · 红 = 不对称\n"
              "★ = 本轮新增或搬迁 · 三条分层规则由测试守着：\n"
              "  ① harness/ 绝不 import swebench/（AST 检查，不是字符串搜索）\n"
              "  ② 「什么算答案」不进执行平面 —— 沙箱归 orchestrator，提取归 benchmark\n"
              "  ③ 粒度决定位置：每次工具调用 → interceptor，每步 → checkpoint，"
              "每个 run → orchestrator")
    els.append(text("legend", 40, 58, legend, size=12, color=GREY))
    top = 58 + text_size(legend, 12)[1] + 40

    # ---------------- row 1: callers | harness runtime ----------------
    cli = panel("cli", 40, top, 340, "调用方（L4）",
                "harness/cli.py —— 只解析参数，不含逻辑\n"
                "  run --transport {stdio,http} ★ --tools <panel> ★\n"
                "      --sandbox-image --backend ★ --runtime-bin ★\n"
                "  batch · show · atif · fork-plan · extract · reap\n"
                "swebench/__main__.py —— 另一个调用方",
                color=BLUE)
    swb = panel("swb", 40, cli["bottom"] + GAP, 340,
                "swebench/ —— 一个插件（L4）",
                "只留「什么算答案」这一件事：\n"
                "  patch.py    答案 = 一个 git diff\n"
                "  AshSession  ★ 现在继承 SandboxSession，\n"
                "              只加 diff 的基线 + get_patch\n"
                "  harnesses/  litellm · claude-code · marathon\n"
                "  configs/    tools: full ★（不是 default）\n"
                "旧路径都留了 shim：backends / templates /\n"
                "agent.tools / mcp_server 都还能 import",
                color=YELLOW)

    hx = max(cli["right"], swb["right"]) + 40
    inner = hx + 16
    orch = panel("orch", inner, top + 34, 640,
                 "orchestrator/run.py —— 一个 run 的一生 ★现在真的是入口",
                 "以前：调用方自己建沙箱和 session 再传进来\n"
                 "      → 只名了个镜像的 run 拿不到任何快照\n"
                 "现在：给个 image，它自己建沙箱、自己起 server、\n"
                 "      自己挂 bridge、逆序拆干净\n"
                 "  RunSpec       transport ★ backend ★ tools ★ image\n"
                 "  OwnedSandbox ★ session + server + pool\n"
                 "     sandbox_id 是记下来的，不是懒读的 ——\n"
                 "     teardown 在组装 outcome 之前跑，销毁后的\n"
                 "     session 只会答 unknown\n"
                 "  batch.py 并发上限 · 重试分类 · 可续跑\n"
                 "  resources.py 写前账本 → harness reap 回收",
                 color=GREEN)
    slots = panel("slots", inner, orch["bottom"] + GAP, 310,
                  "slots/ + normalize/",
                  "claude-code  Agent SDK · hooks · fork_session\n"
                  "codex        SDK over app-server JSON-RPC\n"
                  "opencode     serve HTTP+SSE · 原生 fork\n"
                  "normalize/   原生事件 → journal，纯映射\n"
                  "slot 是黑盒：run / kill / version",
                  color=GREEN)
    rb = panel("rb", slots["right"] + COL, orch["bottom"] + GAP, 310,
               "rollback / extract ★真机验证过",
               "checkpointing.py SnapshotBridge 配对两半\n"
               "  skipped_on_loop ★ 现在有人读（见红框）\n"
               "rollback.py pair = seq ↔ (快照 + 会话 ref)\n"
               "extract.py  事后提取：restore → 提取器 → 销毁\n"
               "core/journal.py 真源；嵌套事件排队分发",
               color=ORANGE)
    gwtop = max(slots["bottom"], rb["bottom"]) + GAP
    gw = panel("gw", inner, gwtop, max(orch["w"], rb["right"] - inner),
               "gateway/ —— 模型缝（已实现且测过，按需接入）",
               "routing.py 模型名 → 任意 endpoint（RL ckpt / vLLM / 供应商）· 每 slot 一个 token\n"
               "server.py  wire tap 落 journal · 预算超了直接 429，不是事后算账",
               color=GREEN)

    # the container is sized from what it actually holds
    l3w = max(orch["right"], rb["right"], gw["right"]) - hx + 16
    l3h = gw["bottom"] - (top) + 16
    els.append(box("l3", hx, top, l3w, l3h, color=GREEN, dashed=True))
    els.append(text("l3-t", hx + PAD, top + 8,
                    "harness/ —— agent 运行时（L3）", size=15, color=GREEN))
    CONTAINS.append(("l3", ["l3-t"]))

    row1_bottom = max(swb["bottom"], top + l3h)

    # ---------------- row 2: the two transports ----------------
    ty = row1_bottom + 46
    els.append(text("tr-t", 40, ty,
                    "★ 本轮核心：沙箱归 orchestrator，transport 只决定 agent 怎么跟它说话",
                    size=15, color=VIOLET))
    sub = "两条路是同一个所有者 —— 它建沙箱、持 handle、拍快照、最后销毁。差别只在「工具调用发生在哪个进程」。"
    els.append(text("tr-b", 40, ty + 26, sub, size=12, color=GREY))
    tby = ty + 26 + text_size(sub, 12)[1] + 18

    http = panel("http", 40, tby, 520, "--transport http ★",
                 "server 在本进程内（socket 先绑好，port=0 也能立刻报 URL）\n"
                 "沙箱用 handle 交出：pool.adopt(sandbox) ★\n"
                 "  entry 标 external：pool 只服务，绝不销毁（销毁归 session）\n"
                 "快门：本进程按 —— 每次改动型调用后\n"
                 "  after_mutating_call → bridge.on_tool_boundary\n"
                 "map 进 journal：同进程，直接写",
                 color=VIOLET)
    stdio = panel("stdio", http["right"] + COL, tby, 500, "--transport stdio",
                  "server 是 slot 自己的子进程：\n"
                  "  --attach <id> --tools <panel> --checkpoint-log <file> ★\n"
                  "沙箱按 id 交出（attach → 只有 microvm，恰好也是\n"
                  "  唯一能拍快照的 backend）\n"
                  "快门：子进程按 ★ —— 边界在它那，串行处理所以必然静止\n"
                  "map 进 journal：每拍一张追加一行 JSONL，run 结束后\n"
                  "  父进程折回 journal，配上会话 ref ★",
                  color=VIOLET)
    asym = panel("asym", 40, max(http["bottom"], stdio["bottom"]) + GAP,
                 stdio["right"] - 40,
                 "原则：checkpoint 挂在工具路径上，谁服务工具调用谁按快门",
                 "turn 边界对 SDK slot 不可用（它在自己的 event loop 里写 journal，进不去 session 的私有 loop）——\n"
                 "以前 bridge 静默跳过、计数没人读，跑完报「成功」而快照是 0。\n"
                 "但工具调用本来就是外部 agent 的一步：它跟环境的唯一通道就是工具调用。\n"
                 "所以快门跟着服务进程走：http 在本进程，stdio 在子进程 —— 回传的只是「第几步→哪个快照」的对照表。\n"
                 "真机同一个 prompt：两条 transport 各 3 个完整 pair；restore 第 1 步快照只见第一个文件，顺序也对。\n"
                 "checkpoint.unavailable 只剩真没边界可站的 wiring 才会报（自己拼 mcp_stdio_args 又不带 --checkpoint-log）。",
                 color=RED)

    # ---------------- row 3: execution plane ----------------
    ey = asym["bottom"] + 46
    inner2 = 56
    sess = panel("sess", inner2, ey + 34, 500,
                 "① session.py ★本轮从 swebench 搬下来",
                 "SandboxSession —— 一个 run 一个沙箱，同步接口\n"
                 "  create / destroy / upload / environment\n"
                 "  snapshot / squash_snapshot / swap_sandbox\n"
                 "     ↑ checkpointing 认的就是这三个方法\n"
                 "  executor_for(agent_id) → (tool, args) → ToolResult\n"
                 "_after_create 钩子 ★：子类在这里记基线；故意不在\n"
                 "  swap_sandbox 后跑 —— 再探一次会把 agent 自己的\n"
                 "  新文件算进基线，从答案里漏掉\n"
                 "templates.py ★ 一起搬下来：按需给每个镜像建 microVM\n"
                 "  模板（裸镜像冷启动起不来 runtime）",
                 color=GREEN)
    pnl = panel("panel", sess["right"] + COL, ey + 34, 500,
                "② panel.py ★本轮从 swebench 搬下来",
                "manifest + runtime/schema/tools.json → 编译出面板\n"
                "  format=openai/anthropic → agent loop 用\n"
                "  format=raw → 就是 MCP 的形状，不需要任何转换\n"
                "搬下来的理由：代理以前手写工具表，跟 runtime 完全不校验\n"
                "harness/tool_panels/ ★\n"
                "  default ★ shell + text_editor —— 不给 background\n"
                "  full · bash_only · no_web\n"
                "swebench 显式钉 full：把跑分工具面从 7 个悄悄缩到 2 个，\n"
                "  会让 results/ 里所有成绩不可比，而没有一行配置变过",
                color=GREEN)
    srv = panel("srv", inner2, max(sess["bottom"], pnl["bottom"]) + GAP, 500,
                "③ server.py —— MCP 代理（stdio | HTTP）",
                "ExecSurface ★ = 面板 + 面板不懂的那部分：sandbox_id\n"
                "  single 视图去掉它（绑定会话 → 模型根本看不到）\n"
                "  multi  视图设为必填（无状态，并发不抢「当前沙箱」）\n"
                "route 在 pipeline / 变更钩子 / 沙箱之前 ★ → 改过名的\n"
                "  视图（run_tests → shell）也能被盯着 shell 的 interceptor 看见\n"
                "start()/stop() ★ 进程内跑 · adopt() ★ 接管已有沙箱\n"
                "身份 X-Session-Owner / Mcp-Session-Id（响应头）\n"
                "绑定 X-Session-Sandbox",
                color=GREEN)
    onion = panel("onion", srv["right"] + COL,
                  max(sess["bottom"], pnl["bottom"]) + GAP, 500,
                  "④ 洋葱 · 何时拍 · 沙箱从哪来",
                  "pipeline.py 四种裁决 Continue/Rewrite/Reject/ShortCircuit\n"
                  "  永不抛异常：上面是 agent loop\n"
                  "interceptors/ 一个文件一个：guardrail · truncate ·\n"
                  "  present · mutation（不是分类法，是刚好需要的）\n"
                  "checkpoints.py 只读步复用上一个快照 · 层链被压缩就 re-board\n"
                  "backends.py docker | microvm | k8s —— 查表，不是实现\n"
                  "  resolve_microvm_endpoint() ★ 一个解析器：以前 pool 读\n"
                  "  AENV_SERVER_URL 而模板构建器不读，同配置一个能用一个不能",
                  color=GREEN)
    l2w = max(pnl["right"], onion["right"]) - 40 + 16
    l2h = max(srv["bottom"], onion["bottom"]) - ey + 16
    els.append(box("l2", 40, ey, l2w, l2h, color=GREEN, dashed=True))
    els.append(text("l2-t", 40 + PAD, ey + 8,
                    "harness/execution/ —— 执行平面（L2）：对「答案」零认知，AST 测试保证不 import swebench",
                    size=15, color=GREEN))
    CONTAINS.append(("l2", ["l2-t"]))

    # ---------------- row 4: L1 + state ----------------
    ly = ey + l2h + 40
    rt = panel("rt", 40, ly, 520, "沙箱（L1）",
               "ash-runtime（Go 静态二进制，跑在沙箱里）\n"
               "  8 个工具：shell · process · text_editor · grep_files\n"
               "    web_fetch · web_search · artifact · wait_for_events\n"
               "  --dump-schema → runtime/schema/tools.json（面板拿它编译）\n"
               "工具集只在 Go 里声明一次，下游全是派生的",
               color=BLUE)
    aenv = panel("aenv", rt["right"] + COL, ly, 520,
                 "AgentENV + firecracker fork",
                 "全量 VM 快照：内存 + vmstate · 增量 overlaybd 层\n"
                 "restore / fork（COW）· pause / resume\n"
                 "★已验证：一个快照分出 3 条分支互不可见；分支的分支两代都在\n"
                 "disk_only 快照会冷启动 → 模板必须声明启动命令，\n"
                 "  否则 restore 出来的沙箱没有 runtime，每个工具调用都 502",
                 color=ORANGE)
    panel("state", 40, max(rt["bottom"], aenv["bottom"]) + GAP,
          aenv["right"] - 40, "状态",
          "791 测试通过 · 7 跳过 · 117 contract 检查\n"
          "真机跑过（Firecracker，共享实例只碰自己建的沙箱）：真 Claude Code 走两条 transport ·\n"
          "  两条各 3 个完整 rollback pair ★ · restore 验证了 map 顺序 · fork_plan 两种 journal 都能读 · 无泄漏\n"
          "还没做：subagent / IAC（先不管）· contracts 活体探针（缺凭据）·\n"
          "  CLI slot 对 microvm 的 fork 端到端",
          color=ORANGE)

    # ---------------- arrows ----------------
    els.append(arrow("a1", cli["right"] + 4, cli["y"] + 40,
                     [[0, 0], [hx - cli["right"] - 8, 0]]))
    els.append(text("a1l", cli["right"] + 8, cli["y"] + 16, "RunSpec",
                    size=11, color=GREY))
    els.append(arrow("a2", swb["right"] + 4, swb["y"] + 60,
                     [[0, 0], [hx - swb["right"] - 8, 0]], dashed=True))
    els.append(text("a2l", swb["right"] + 6, swb["y"] + 34, "提取器",
                    size=11, color=GREY))
    els.append(arrow("a3", 300, tby - 14, [[0, 0], [0, 12]], color=VIOLET))
    els.append(arrow("a4", stdio["x"] + 260, tby - 14, [[0, 0], [0, 12]],
                     color=VIOLET))
    els.append(arrow("a5", 300, asym["y"] - 14, [[0, 0], [0, 12]], color=RED))
    els.append(arrow("a6", stdio["x"] + 260, asym["y"] - 14, [[0, 0], [0, 12]],
                     color=RED))
    els.append(arrow("a7", 400, ly - 34, [[0, 0], [0, 28]]))
    els.append(text("a7l", 410, ly - 30, "JSON-RPC / build_pool", size=11,
                    color=GREY))


# --- validation ------------------------------------------------------------
def _rects(kinds=("rectangle",)) -> list:
    return [e for e in els if e["type"] in kinds]


def validate() -> list:
    """Every geometric mistake the hand-written version actually made."""
    problems = []
    by_id = {e["id"]: e for e in els}

    # 1. text must fit inside the box it is declared to live in
    for box_id, text_ids in CONTAINS:
        b = by_id[box_id]
        for tid in text_ids:
            t = by_id[tid]
            if t["x"] < b["x"] or t["y"] < b["y"]:
                problems.append("%s starts before its box %s" % (tid, box_id))
            if t["x"] + t["width"] > b["x"] + b["width"] - 4:
                problems.append("%s overflows %s horizontally by %.0fpx"
                                % (tid, box_id,
                                   t["x"] + t["width"] - (b["x"] + b["width"])))
            if t["y"] + t["height"] > b["y"] + b["height"] - 2:
                problems.append("%s overflows %s vertically by %.0fpx"
                                % (tid, box_id,
                                   t["y"] + t["height"] - (b["y"] + b["height"])))

    # 2. panel boxes must not overlap each other (a container may hold panels)
    containers = {"l2", "l3"}
    panels = [b for b in _rects() if b["id"] not in containers]
    for i, a in enumerate(panels):
        for b in panels[i + 1:]:
            if (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
                    and a["y"] < b["y"] + b["height"]
                    and b["y"] < a["y"] + a["height"]):
                problems.append("boxes %s and %s overlap" % (a["id"], b["id"]))

    # 3. free-standing text must not collide with another free-standing text
    contained = {t for _, ts in CONTAINS for t in ts}
    loose = [e for e in els
             if e["type"] == "text" and e["id"] not in contained]
    for i, a in enumerate(loose):
        for b in loose[i + 1:]:
            if (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
                    and a["y"] < b["y"] + b["height"]
                    and b["y"] < a["y"] + a["height"]):
                problems.append("labels %s and %s collide" % (a["id"], b["id"]))

    # 4. every element inside a declared container really is inside it
    for cid in containers:
        c = by_id.get(cid)
        if c is None:
            continue
        for e in _rects():
            if e["id"] in containers or e["id"] == cid:
                continue
            inside_x = c["x"] <= e["x"] and e["x"] + e["width"] <= c["x"] + c["width"]
            straddles_y = (c["y"] < e["y"] + e["height"]
                           and e["y"] < c["y"] + c["height"])
            if straddles_y and inside_x:
                fits = e["y"] + e["height"] <= c["y"] + c["height"]
                if not fits:
                    problems.append("%s escapes container %s vertically"
                                    % (e["id"], cid))
    return problems


def main() -> int:
    build()
    problems = validate()
    if problems:
        print("REFUSING to write -- %d geometry problem(s):" % len(problems))
        for p in problems:
            print("  -", p)
        return 1
    document = {
        "type": "excalidraw", "version": 2, "source": "claude",
        "appState": {"gridSize": None, "viewBackgroundColor": "#121212"},
        "files": {}, "elements": els,
    }
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=1))
    print("wrote %s (%d elements, geometry checks passed)" % (OUT, len(els)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
