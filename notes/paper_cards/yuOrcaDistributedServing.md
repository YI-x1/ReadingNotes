# Citation Key

`yuOrcaDistributedServing`

文献：Orca: A Distributed Serving System for Transformer-Based Generative Models

# One Sentence Summary

Orca 提出 iteration-level scheduling 和 selective batching，让生成式 Transformer 服务可以在每轮 decoding 后动态加入和移除请求，从而显著提高 batching 效率。

# Research Question

传统 serving 系统常以 request 为单位调度，整个生成过程占住 batch slot，导致短请求提前完成后资源浪费、长请求阻塞新请求。论文要回答的是：能否把调度粒度降到每次 decoding iteration，并仍保持 Transformer 执行高效？

# Method

Orca 的调度器每次只调度一次迭代。请求完成时可以立即离开 batch，新请求可以在下一轮进入，从而避免 request-level batching 的空洞。

Selective batching 则区分不同算子：对参数密集、形状兼容的操作继续批处理；对 attention 中与各请求历史长度相关的部分做拆分处理。系统还支持模型并行和分布式执行。

# Dataset

实验使用 synthetic traces 和不同 GPT 配置，包括 13B、101B、175B、341B 等规模；到达过程和输入/输出长度分布用于模拟在线服务负载，并与 FasterTransformer 类基线对比。

# Main Findings

Orca 在相同延迟约束下对 GPT-3 175B 类模型可实现最高约 36.9x 吞吐提升。核心收益来自 iteration-level scheduling 消除了 early-finish/late-join 的 batch 低效。

论文还显示 selective batching 能在支持不同序列长度的同时保持底层执行效率。

# Contribution

- 将生成式模型服务的调度粒度从 request-level 降到 iteration-level。
- 提出 selective batching，解决自回归生成中不同请求历史长度不一致的问题。
- 给出面向大模型的分布式 serving 架构。

# Limitations

Orca 没有从根本上解决 KV cache 的显存碎片和分页管理问题，这正是 PagedAttention/vLLM 后来重点处理的方向。

实验 workload 偏合成，真实生产分布、prefix 共享、结构化输出程序等场景展开有限。

# Relation To My Research

Orca 是 LLM serving 调度史上的关键节点。它解释了为什么 decode 阶段要按 iteration 调度，也为后来的 vLLM continuous batching、PagedAttention 和 SGLang runtime 打下了系统语境。

# Useful Quotes

- §1 可引用观点：自回归生成的每个请求输出长度不同，request-level batching 会造成资源浪费。
- §3 可引用观点：iteration-level scheduling 允许请求在每轮生成后进入或退出 batch。
- §4 可引用观点：selective batching 让不同长度请求仍能共享参数密集计算。

# Critical Notes

- Orca 的强项是调度粒度，不是显存管理；和 PagedAttention 对读会更完整。
- 它把 serving 问题从“单个请求跑得快”转成“动态请求池整体跑得稳”。
- 后续系统很多所谓 continuous batching 的思想，都能追溯到这篇。
