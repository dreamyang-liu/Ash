# TVCACHE 官方实现核实（老师质询的回复）

> 老师会上两次要求 check：① TVCACHE 是不是梦阳说的"每次 tool call 生成 mutation 判断的稀疏采样"？② 如果是更灵活的匹配，为什么 lookup 反而比 exact 慢？
>
> 结论先行：**梦阳的猜测不成立**（对我们跑的 TCG 配置而言），TVCACHE 的灵活性来自"允许命中浅于当前步数的前缀 checkpoint"，与环境语义无关。

## 1. 我们跑的是官方哪个类、怎么调的

- 官方仓库 [TVCache/TVCache](https://github.com/TVCache/TVCache)，revision `3a4f95a6582eea1e4b7e84a9f3a0c74eaa8fde02`，零改动。
- 类：`tvcache/server/tvcache/immutable_env_prefix_tree.py::ImmutableEnvPrefixTreeCache`
- 写入：`put(task, tool_calls_prefix, env_id, values, tool_exec_times)` —— 每个 source checkpoint（step 1..k 的完整调用前缀）插入 trie。
- 查询：`prefix_match(task, tool_calls)` —— 沿 trie 逐条匹配，返回最深的有 `env_id` 的节点。

## 2. 它到底做了什么匹配（读源码结论）

`ImmutableEnvPrefixTreeCache` 是一棵 **以"逐条 tool call 序列化文本"为边的前缀树（trie）**：

- 匹配 = **字面前缀匹配**。query 的调用序列必须与 source 的调用序列逐字相同，才能走到对应深度。
- **没有任何环境状态语义**：没有 mutation 判定、没有 workspace 感知。仓库里确有 `StatelessCommands` / `mark_stateless`（手工把某条调用标记为无状态）以及 `MutableEnvPrefixTree`（另一棵树），但：
  - `mark_stateless` 需要**外部显式调用**才会打标，我们的评估（忠实复现）没有调用它；
  - `sintel_prefix_match` / `intel_prefix_match` 在 immutable 版里是 `raise NotImplementedError`；
  - 我们跑的 `put + prefix_match` 路径完全不含这些机制。
- 所以对本次实验配置：**TVCACHE TCG ≈ exact tool-call sequence matching + 允许部分前缀命中**。梦阳说的"每步判 mutation 的稀疏采样"属于 TVCACHE 论文中更完整的形态，但不在我们复现的这条代码路径里。

## 3. 回答老师的两个疑问

### ① 为什么"更灵活"反而命中率更高（8/248 vs exact 2/248）？

灵活性不是"匹配得更聪明"，而是**命中判定的位置不同**：

- 我们的 Exact Full History：query 第 i 步的查询 = "是否存在 source 前缀 1..i 与 query 前缀 1..i **完全相等**"。
- TVCACHE：query 第 i 步的查询 = "最长公共前缀深度 d ≤ i 处是否有已注册环境"，**允许 d < i**。即 query 走了 4 步、其中前 3 步与 source 逐字相同时，TVCACHE 命中 source 第 3 步的 checkpoint；exact 在第 4 步直接 miss。
- 这 6 个额外命中（8-2）全部来自"前 k 步相同、随后分叉"的场景。

### ② 为什么 lookup 反而比 exact 慢（0.0017ms vs 0.0003ms）？

- exact：一次规范化 + 一次 dict 哈希查表。
- TVCACHE：逐条 call 走树、Python dict 查找、每次加 `threading.Lock`、还有 TTL 过期检查。
- 两者都是微秒级；老师的判断正确——**该开销相对 rollout 完全 trivial，不构成选型因素**，此前表述"它更快"并不成立，已更正。

## 4. 为什么 TVCACHE 远低于 Ours（8/248 vs 51/248）

TVCACHE 匹配的是**调用文本**，而跨 policy rollout（Luna ↔ Qwen）对同一目标会用不同的探索命令（`ls -la /app` vs `find /app ...`）。这些差异对环境无影响，但字面匹配全部 miss。Ours 的 relaxed projection 把可证明的只读命令从 state key 中剔除，因此能跨 policy 命中。这正是我们方法的设计目标，也是两者的本质区别：**TVCACHE 匹配"做过什么"，Ours 匹配"环境变成了什么"**。

## 5. 复现忠实性

- 使用官方 repo、官方 `ImmutableEnvPrefixTreeCache`、官方 `CACHE_BUDGET=100000`（关闭容量驱逐干扰）、`put`/`prefix_match` 官方语义，未做任何 patch。
- 若老师认为 TVCACHE 论文中的完整形态（含 stateless 标注 / intel 模式）应作为 baseline，那属于另一条代码路径（部分未在官方仓库实现），需要与原作者确认后再补跑——我们当前报告的数字严格对应官方可运行实现。

— 2026-09-06，依据 `/tmp/TVCache`（官方 revision 3a4f95a）源码逐行核实。
