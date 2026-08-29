#!/usr/bin/env python3
"""The layered-architecture half of docs/architecture-overview.excalidraw.

Not a diagram of its own any more: it rendered a strict subset of the overview
(0 text blocks that the overview lacked), so the two were one picture stored
twice -- 56KB of generated JSON that could only drift. ``build()`` is imported by
gen_overview.py, which draws this above its sequence diagram.

Checked in because the previous diagram was hand-written JSON, which went wrong in
two ways worth avoiding: typos that made the file unopenable, and -- worse -- text
that silently overflowed its box or boxes that overlapped, so the picture looked
fine to whoever wrote it and was unreadable to whoever opened it.

Everything is laid out from measured geometry, and exgen's validator refuses to
write a file whose boxes overlap, whose text escapes its box, or whose labels sit
on top of each other. Run it after changing the architecture:

    python3 docs/gen_architecture.py

Font metrics are approximations of Excalidraw's default (Virgil/Excalifont) at the
sizes used here, deliberately pessimistic: a CJK glyph is a full em, latin ~0.58,
so the checker over-estimates width rather than passing something that will clip.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Only used when this file is run directly, as a way to check the layout of the
#: architecture half alone. The committed artefact is architecture-overview.
OUT = Path(__file__).resolve().parent / "architecture-only.excalidraw"

# The element builders, font metrics and geometry validation live in exgen.py,
# shared with gen_checkpoint_flow.py -- two copies of the validator is how the
# two diagrams would come to disagree about what "fits" means.
from exgen import (BLUE, GREEN, GREY, ORANGE, PAD, RED, VIOLET,  # noqa: E402
                   WHITE, YELLOW, Canvas, text_size)

_c = Canvas()


def build() -> None:
    """Lay out top to bottom, each row from the previous row's measured bottom.

    Nothing here hard-codes a coordinate that depends on how tall some other box
    turned out to be. The previous diagram did, which is why a one-line edit to a
    label could push a box out of its container with no visible sign.
    """
    GAP = 16
    COL = 20          # gap between columns
    _c.text("title", 40, 24,
                    "Ash 架构（feat/harness-v1，本轮改动后）", size=20, color=BLUE)
    legend = ("绿 = 执行平面 · 蓝 = 调用方 · 橙 = 状态 · 黄 = benchmark 插件 · "
              "紫 = 两条 transport · 红 = 不对称\n"
              "★ = 本轮新增或搬迁 · 三条分层规则由测试守着：\n"
              "  ① harness/ 绝不 import swebench/（AST 检查，不是字符串搜索）\n"
              "  ② 「什么算答案」不进执行平面 —— 沙箱归 orchestrator，提取归 benchmark\n"
              "  ③ 粒度决定位置：每次工具调用 → interceptor，每步 → checkpoint，"
              "每个 run → orchestrator")
    _c.text("legend", 40, 58, legend, size=12, color=GREY)
    top = 58 + text_size(legend, 12)[1] + 40

    # ---------------- row 1: callers | harness runtime ----------------
    cli = _c.panel("cli", 40, top, 340, "调用方（L4）",
                "harness/cli.py —— 只解析参数，不含逻辑\n"
                "  run --transport {stdio,http} ★ --tools <panel> ★\n"
                "      --sandbox-image --backend ★ --runtime-bin ★\n"
                "  batch · show · atif · fork-plan · extract · reap\n"
                "swebench/__main__.py —— 另一个调用方",
                color=BLUE)
    swb = _c.panel("swb", 40, cli["bottom"] + GAP, 340,
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
    orch = _c.panel("orch", inner, top + 34, 640,
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
    slots = _c.panel("slots", inner, orch["bottom"] + GAP, 310,
                  "slots/ + normalize/",
                  "claude-code  Agent SDK · hooks · fork_session\n"
                  "codex        SDK over app-server JSON-RPC\n"
                  "opencode     serve HTTP+SSE · 原生 fork\n"
                  "normalize/   原生事件 → journal，纯映射\n"
                  "slot 是黑盒：run / kill / version",
                  color=GREEN)
    rb = _c.panel("rb", slots["right"] + COL, orch["bottom"] + GAP, 310,
               "rollback / extract ★真机验证过",
               "checkpointing.py SnapshotBridge 配对两半\n"
               "  skipped_on_loop ★ 现在有人读（见红框）\n"
               "rollback.py pair = seq ↔ (快照 + 会话 ref)\n"
               "extract.py  事后提取：restore → 提取器 → 销毁\n"
               "core/journal.py 真源；嵌套事件排队分发",
               color=ORANGE)
    gwtop = max(slots["bottom"], rb["bottom"]) + GAP
    gw = _c.panel("gw", inner, gwtop, max(orch["w"], rb["right"] - inner),
               "gateway/ —— 模型缝（已实现且测过，按需接入）",
               "routing.py 模型名 → 任意 endpoint（RL ckpt / vLLM / 供应商）· 每 slot 一个 token\n"
               "server.py  wire tap 落 journal · 预算超了直接 429，不是事后算账",
               color=GREEN)

    # the container is sized from what it actually holds
    l3w = max(orch["right"], rb["right"], gw["right"]) - hx + 16
    l3h = gw["bottom"] - (top) + 16
    _c.box("l3", hx, top, l3w, l3h, color=GREEN, dashed=True)
    _c.text("l3-t", hx + PAD, top + 8,
                    "harness/ —— agent 运行时（L3）", size=15, color=GREEN)
    _c.contains.append(("l3", ["l3-t"]))

    row1_bottom = max(swb["bottom"], top + l3h)

    # ---------------- row 2: the two transports ----------------
    ty = row1_bottom + 46
    _c.text("tr-t", 40, ty,
                    "★ 本轮核心：沙箱归 orchestrator，transport 只决定 agent 怎么跟它说话",
                    size=15, color=VIOLET)
    sub = "两条路是同一个所有者 —— 它建沙箱、持 handle、拍快照、最后销毁。差别只在「工具调用发生在哪个进程」。"
    _c.text("tr-b", 40, ty + 26, sub, size=12, color=GREY)
    tby = ty + 26 + text_size(sub, 12)[1] + 18

    http = _c.panel("http", 40, tby, 520, "--transport http ★",
                 "server 在本进程内（socket 先绑好，port=0 也能立刻报 URL）\n"
                 "沙箱用 handle 交出：pool.adopt(sandbox) ★\n"
                 "  entry 标 external：pool 只服务，绝不销毁（销毁归 session）\n"
                 "checkpoint 机制由 orchestrator 构造：\n"
                 "  tracker 挂 server 的 pipeline，bridge 直接写 journal",
                 color=VIOLET)
    stdio = _c.panel("stdio", http["right"] + COL, tby, 500, "--transport stdio",
                  "server 是 slot 自己的子进程：\n"
                  "  --attach <id> --tools <panel> --checkpoint-log <file>\n"
                  "沙箱按 id 交出（attach → 只有 microvm，恰好也是\n"
                  "  唯一能拍快照的 backend）\n"
                  "checkpoint 机制由 server main 构造：\n"
                  "  同样的 tracker + Checkpointer，记录写 JSONL，\n"
                  "  run 结束后父进程折回 journal、配上会话 ref",
                  color=VIOLET)
    asym = _c.panel("asym", 40, max(http["bottom"], stdio["bottom"]) + GAP,
                 stdio["right"] - 40,
                 "一套机制 ★：MutationTracker（interceptor）+ Checkpointer，挂在服务工具调用的那条 pipeline 上",
                 "tracker 判断这步改没改（view/grep 不算），Checkpointer 每个 exec 调用后跑一次：\n"
                 "  改了 → 拍新快照；没改 → map 指向上一张（不花钱，map 依然完整）；层链压缩 → re-board；128 层 → squash\n"
                 "两条 transport 只差「谁构造」和「记录落哪」。之前是两套散装 mixin，两边各错一半：\n"
                 "  http 的 tracker 没挂上 pipeline、没人喂 → 不开 always 时每一步都被判「clean」，agent 写了文件 map 却说没变\n"
                 "  stdio 手写捕获 → view 白拍一张、grep 那步没有 map 条目、也没有层链维护\n"
                 "真机验证（不开 always）：write→拍、view→clean 复用、write→再拍，两条 transport 逐条一致。",
                 color=RED)

    # ---------------- row 3: execution plane ----------------
    ey = asym["bottom"] + 46
    inner2 = 56
    sess = _c.panel("sess", inner2, ey + 34, 500,
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
    pnl = _c.panel("panel", sess["right"] + COL, ey + 34, 500,
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
    srv = _c.panel("srv", inner2, max(sess["bottom"], pnl["bottom"]) + GAP, 500,
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
    onion = _c.panel("onion", srv["right"] + COL,
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
    _c.box("l2", 40, ey, l2w, l2h, color=GREEN, dashed=True)
    _c.text("l2-t", 40 + PAD, ey + 8,
                    "harness/execution/ —— 执行平面（L2）：对「答案」零认知，AST 测试保证不 import swebench",
                    size=15, color=GREEN)
    _c.contains.append(("l2", ["l2-t"]))

    # ---------------- row 4: L1 + state ----------------
    ly = ey + l2h + 40
    rt = _c.panel("rt", 40, ly, 520, "沙箱（L1）",
               "ash-runtime（Go 静态二进制，跑在沙箱里）\n"
               "  8 个工具：shell · process · text_editor · grep_files\n"
               "    web_fetch · web_search · artifact · wait_for_events\n"
               "  --dump-schema → runtime/schema/tools.json（面板拿它编译）\n"
               "工具集只在 Go 里声明一次，下游全是派生的",
               color=BLUE)
    aenv = _c.panel("aenv", rt["right"] + COL, ly, 520,
                 "AgentENV + firecracker fork",
                 "全量 VM 快照：内存 + vmstate · 增量 overlaybd 层\n"
                 "restore / fork（COW）· pause / resume\n"
                 "★已验证：一个快照分出 3 条分支互不可见；分支的分支两代都在\n"
                 "disk_only 快照会冷启动 → 模板必须声明启动命令，\n"
                 "  否则 restore 出来的沙箱没有 runtime，每个工具调用都 502",
                 color=ORANGE)
    _c.panel("state", 40, max(rt["bottom"], aenv["bottom"]) + GAP,
          aenv["right"] - 40, "状态",
          "793 测试通过 · 7 跳过 · 117 contract 检查\n"
          "真机跑过（Firecracker，共享实例只碰自己建的沙箱）：真 Claude Code 走两条 transport ·\n"
          "  两条各 3 个完整 rollback pair · capture/clean 逐条一致 · restore 验证了 map 顺序 · 无泄漏\n"
          "SWE-bench Verified 真实实例过链 ★：每个改动步一对 pair · 新 microVM 从任一步恢复 ·\n"
          "  第 2 步分岔出「验证原修法(45 测试过)」和「推翻重写」两个隔离分支,最终 diff 各异\n"
          "还没做：subagent / IAC（先不管）· contracts 活体探针（缺凭据）·\n"
          "  CLI slot 对 microvm 的 fork 端到端",
          color=ORANGE)

    # ---------------- arrows ----------------
    _c.arrow("a1", cli["right"] + 4, cli["y"] + 40,
                     [[0, 0], [hx - cli["right"] - 8, 0]])
    _c.text("a1l", cli["right"] + 8, cli["y"] + 16, "RunSpec",
                    size=11, color=GREY)
    _c.arrow("a2", swb["right"] + 4, swb["y"] + 60,
                     [[0, 0], [hx - swb["right"] - 8, 0]], dashed=True)
    _c.text("a2l", swb["right"] + 6, swb["y"] + 34, "提取器",
                    size=11, color=GREY)
    _c.arrow("a3", 300, tby - 14, [[0, 0], [0, 12]], color=VIOLET)
    _c.arrow("a4", stdio["x"] + 260, tby - 14, [[0, 0], [0, 12]],
                     color=VIOLET)
    _c.arrow("a5", 300, asym["y"] - 14, [[0, 0], [0, 12]], color=RED)
    _c.arrow("a6", stdio["x"] + 260, asym["y"] - 14, [[0, 0], [0, 12]],
                     color=RED)
    _c.arrow("a7", 400, ly - 34, [[0, 0], [0, 28]])
    _c.text("a7l", 410, ly - 30, "JSON-RPC / build_pool", size=11,
                    color=GREY)


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
    # l2/l3 are containers: panels legitimately sit inside them.
    _c.write(OUT, containers={"l2", "l3"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
