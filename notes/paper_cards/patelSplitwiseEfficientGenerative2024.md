# Citation Key

`patelSplitwiseEfficientGenerative2024`

文献：Splitwise: Efficient Generative LLM Inference Using Phase Splitting

# One Sentence Summary

Splitwise 将 prompt processing 和 token generation 分配到不同机器池，利用两阶段在计算强度、显存需求和功耗上的差异来提高吞吐并降低成本。

# Research Question

生成式 LLM 推理由 prompt 阶段和 token 阶段组成。论文要回答的是：如果两阶段的资源瓶颈不同，是否应使用不同硬件和调度策略，而不是在同一批机器上混合执行？

# Method

Splitwise 设计 prompt pool、token pool 和 mixed pool。Prompt 机器处理输入 prompt、生成第一个 token 和 KV cache，然后通过高速网络把 KV cache 交给 token 机器继续自回归生成。

系统根据 SLO、机器类型、功耗和成本探索不同集群配置。核心思想是让 compute-heavy 的 prompt 阶段使用更适合计算的配置，让 memory-heavy 的 token 阶段使用更适合显存容量/带宽的配置。

# Dataset

论文使用来自 Azure 的生产 workload traces，包括代码生成和对话场景；分析 prompt length、output length、TTFT、time between tokens、端到端延迟和吞吐。

# Main Findings

生产 trace 显示 prompt 和 token 阶段长度分布差异明显，token generation 往往主导端到端延迟。Prompt 阶段更 compute-bound，token 阶段更受显存容量和带宽限制。

Splitwise 在满足 SLO 的前提下可获得更高吞吐和更低成本；论文报告了约 1.4x 吞吐提升同时降低成本，以及在等成本/功耗场景下约 2.35x 吞吐提升。

# Contribution

- 用生产 traces 量化生成式 LLM 两阶段的资源差异。
- 提出 phase splitting 的服务架构，将 prompt 与 token generation 分配到不同机器池。
- 从成本、功耗和 SLO 角度讨论异构集群设计。

# Limitations

Splitwise 依赖高速网络传输 KV cache，并需要精细调度来避免两阶段资源不匹配。不同业务的 output 长度、batching 策略和 SLO 会改变最优机器配比。

相比 DistServe，论文更偏资源和成本建模，对复杂 placement、goodput SLO 达成率和细粒度动态调度的展开较少。

# Relation To My Research

Splitwise 是理解 prefill/decode disaggregation 的重要系统论文，和 DistServe 构成一组：Splitwise 强调异构硬件与成本，DistServe 强调 goodput 和 placement。它也能解释为什么 LLM serving 不能只优化单个 attention kernel。

# Useful Quotes

- Abstract 可引用观点：prompt computation 和 token generation 适合不同硬件，因为一个偏计算密集，一个偏内存密集。
- §2 可引用观点：token generation 经常占据端到端延迟的大部分。
- §4 可引用观点：把 KV cache 从 prompt 机器传给 token 机器，是 phase splitting 能成立的系统连接点。

# Critical Notes

- 这篇论文的实用价值在于生产 trace 分析：它把“阶段异质性”从直觉变成了部署决策依据。
- 与 DistServe 对比时要看目标函数差异：成本/功耗优化 vs goodput/SLO 优化。
- 对实现者来说，最重要的问题是两阶段队列如何稳定配平，否则拆分会把瓶颈从 GPU 迁到调度器。
