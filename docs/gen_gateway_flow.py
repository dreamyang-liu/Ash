#!/usr/bin/env python3
"""Generate docs/gateway-flow.excalidraw — one request through the gateway.

The full lifecycle: model configuration (routes.json) -> orchestrator setup
(mint token, env injection) -> the agent's request -> what the gateway does to
it, in order -> the upstream -> what comes back and what gets recorded. Every
step is the code's actual order (server.py do_POST / _forward / _tap).

    python3 docs/gen_gateway_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exgen import BLUE, GREEN, GREY, ORANGE, RED, VIOLET, Canvas, text_size  # noqa: E402
from gen_overview import Sequence  # noqa: E402

OUT = Path(__file__).resolve().parent / "gateway-flow.excalidraw"


def main() -> int:
    c = Canvas()
    c.text("title", 40, 24,
           "Inference gateway:从配置 model 到一个 request 真正发出去",
           size=20, color=BLUE)
    legend = ("routes.json 是唯一的模型配置点;agent 拿到的只有 gateway 地址 + slot-token。\n"
              "实线 = 请求/数据 · 虚线 = 记录 · 紫框 = 每个 LLM 请求都重复的部分")
    c.text("legend", 40, 56, legend, size=12, color=GREY)
    top = 56 + text_size(legend, 12)[1] + 22

    # ---- who can go through the gateway, per agent --------------------------
    cc = c.panel("ag-cc", 40, top, 400, "claude-code —— 过 gateway ✅(已真机验证)",
                 "说 Anthropic Messages 协议,正是 gateway 说的\n"
                 "接法:env 注入 ANTHROPIC_BASE_URL + AUTH_TOKEN\n"
                 "slot 自动关 Bedrock/Vertex 直连(否则静默绕过)\n"
                 "已验:模型改写 · 计价 · 预算 400 · wire-tap",
                 color=GREEN)
    oc = c.panel("ag-oc", cc["right"] + 18, top, 400,
                 "opencode —— 协议匹配,未真机验证",
                 "用它的 anthropic provider 时也说 /v1/messages\n"
                 "接法:opencode 配置里 provider.options.baseURL\n"
                 "  指到 gateway,api key 填 slot-token\n"
                 "理论上直接能用;跑过之前不算数",
                 color=ORANGE)
    cx = c.panel("ag-cx", oc["right"] + 18, top, 400,
                 "codex —— 过不了 gateway ❌(协议不同)",
                 "只说 OpenAI Responses 协议(0.145 起 chat 也不说了)\n"
                 "gateway 只会 Anthropic Messages → 接不上\n"
                 "现状:provider-direct(原生 amazon-bedrock / ChatGPT)\n"
                 "  → 预算和记账对 codex 不生效,这是已知缺口\n"
                 "要接入:gateway 学 Responses 协议(等真需要再做)",
                 color=RED)
    bar_bottom = max(cc["bottom"], oc["bottom"], cx["bottom"])
    c.text("seq-cap", 40, bar_bottom + 18,
           "下面的时序对「说 Anthropic 协议的 agent」成立(claude-code 已验,opencode 待验):",
           size=13, color=GREY)

    seq = Sequence(c, 40, bar_bottom + 44, [
        ("cfg", "routes.json\n(模型→上游的表)", ORANGE),
        ("orch", "Orchestrator", GREEN),
        ("agent", "Agent\n(claude✅ / opencode?)", BLUE),
        ("gw", "Gateway\n(进程内 HTTP)", GREEN),
        ("up", "上游\n(provider/vLLM/ckpt)", VIOLET),
        ("journal", "Journal", ORANGE),
    ], col_w=215)

    # ---- setup ---------------------------------------------------------------
    seq.message("cfg", "orch", 'RoutingTable.from_file:每个 model 名 →'
                               ' {base_url, 真 api_key, 改写名, 单价}')
    seq.message("orch", "gw", "起 GatewayServer(临时端口)+ mint slot-token{agent_id, budget}")
    seq.message("orch", "agent", "env 注入:ANTHROPIC_BASE_URL=gateway · AUTH_TOKEN=slot-token")
    seq.note("   slot 看到 base_url → 自动关 Bedrock/Vertex 直连(否则流量绕过 gateway,实测踩过)",
             color=RED)
    seq.message("agent", "gw", "启动探活 GET /v1/models(失败即 CLI 拒启,gateway 自己应答)")

    # ---- per request ----------------------------------------------------------
    loop = seq.frame_start()
    seq.message("agent", "gw", "POST /v1/messages{model, messages, stream} + Bearer slot-token")
    seq.message("gw", "gw", "① 认token→agent_id ② 查预算:超了→400 不可重试,run 当场死")
    seq.message("gw", "gw", "③ 查路由(model名精确匹配,否则default) ④ 改写:model名换、真key换上")
    seq.message("gw", "up", "转发 base_url+/v1/messages(agent 的 slot-token 永远不出 gateway)")
    seq.message("up", "gw", "SSE 流式响应")
    seq.message("gw", "agent", "逐块透传(零缓冲;响应字节不改——thinking 签名必须逐字节存活)")
    seq.message("gw", "gw", "⑤ 旁听器从 message_start/delta 捞 usage ⑥ usage×单价→spent累加")
    seq.message("gw", "journal", "gateway.request{要的model, 实际model, usage, cost, spent}",
                dashed=True)
    seq.frame("gw-loop", loop, "循环:agent 的每一个 LLM 请求(预算在请求之间查——在飞的那条会跑完)")

    seq.note("换模型 = routes.json 改一行(agent 要 opus-5,上游跑 sonnet-4-6,实测无感);"
             "RL rollout = base_url 指向自己的 checkpoint", color=VIOLET)
    seq.note("codex 不在这张时序里:它的流量今天走 provider-direct,不经过任何一个框——"
             "统一预算/记账要覆盖 codex,先给 gateway 加 Responses 协议", color=RED)
    seq.note("路由没配单价而又设了 budget → journal 记一次 budget_unenforceable,"
             "绝不静默假装在管账", color=RED)
    seq.lifelines()

    c.write(OUT, containers={"gw-loop"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
