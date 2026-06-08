# Citation Key

`DistServeDisaggregatingPrefill`

文献：DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

# One Sentence Summary

DistServe 把 LLM 推理的 prefill 和 decoding 阶段拆到不同 GPU 组上运行，以解除两阶段在算力、显存带宽、并行策略和延迟目标上的相互干扰，从而优化 goodput。

# Research Question

在线 LLM 服务同时关心 TTFT 和 TPOT。论文要回答的是：如果 prefill 与 decoding 的资源需求完全不同，继续把它们 colocate 在同一 GPU 上是否会限制吞吐；能否通过阶段拆分获得更高 SLO 内吞吐？

# Method

DistServe 将请求拆成 prefill 与 decoding 两个阶段。Prefill 实例负责处理 prompt、生成首 token 和 KV cache；decoding 实例负责后续自回归生成。两阶段之间通过网络传输 KV cache。

系统进一步做三类调度决策：为 prefill/decoding 分配不同数量的 GPU；为两阶段选择不同并行策略；在多节点环境下进行带宽感知 placement，尽量降低 KV cache 传输带来的延迟。

# Dataset

实验使用 4 个节点、32 张 A100-80GB GPU，模型包括 OPT 系列，负载包括 ShareGPT 聊天、HumanEval 代码生成和 LongBench 摘要等不同 prompt/output 分布。

# Main Findings

DistServe 在 ShareGPT 上相比 colocated serving 可支持约 2.0x 到 3.41x 更高请求率，在代码生成上约 3.2x，在摘要场景最高约 4.48x。论文还报告了在某些设置下可满足约 10.2x 更严格的 SLO。

一个重要观察是：KV cache 传输并不必然成为瓶颈。只要 placement 合理，传输开销相对端到端延迟可以很小。

# Contribution

- 明确把 LLM serving 的目标从 raw throughput 转为 goodput，即满足延迟 SLO 的有效吞吐。
- 系统化揭示 prefill compute-bound、decoding memory-bandwidth-bound 的阶段差异。
- 给出 prefill-decoding disaggregation 的资源分配、并行策略和 placement 方法。

# Limitations

系统收益依赖网络带宽、模型大小、请求长度分布和 SLO 设置。若 prompt 很短、output 很短，或者跨节点 KV cache 传输很差，拆分收益会下降。

系统实现复杂度较高，需要额外的 KV cache 生命周期管理、跨实例路由和动态资源配置。

# Relation To My Research

DistServe 是从“单 kernel 优化”走向“端到端 serving 架构”的关键论文。它与 Splitwise 强相关，二者都利用 prefill/decode 异质性；也与 PagedAttention、Orca、SGLang 互补，分别解决 KV 管理、调度粒度和程序级执行复用问题。

# Useful Quotes

- Abstract 可引用观点：colocated serving 会让 prefill 和 decoding 互相干扰，并把两阶段绑定到相同资源配置。
- §2 可引用观点：prefill 更接近 compute-bound，decoding 更接近 memory-bandwidth-bound。
- §5 可引用观点：goodput 应以 TTFT/TPOT SLO 达成率为约束，而不是只看每秒 token。

# Critical Notes

- 这篇论文非常适合和 Splitwise 对读：Splitwise 更强调异构硬件和成本，DistServe 更强调 goodput、placement 和服务质量。
- 对实际部署而言，最难的部分可能不是 KV 传输，而是让调度器稳定地适应负载分布变化。
- 需要注意它对 workload 的假设：不同业务的 prompt/output 长度分布会显著影响最优 prefill/decode 配比。
