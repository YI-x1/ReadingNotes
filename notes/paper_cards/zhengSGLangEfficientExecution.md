# Citation Key

`zhengSGLangEfficientExecution`

文献：SGLang: Efficient Execution of Structured Language Model Programs

# One Sentence Summary

SGLang 同时设计前端 DSL 和后端 runtime，用 RadixAttention、cache-aware scheduling 和 compressed FSM 加速多轮、多分支、结构化的语言模型程序。

# Research Question

许多 LLM 应用不只是一次 prompt 生成，而是包含多轮调用、分支、工具格式、few-shot 示例和结构化约束。论文要回答的是：能否让这些程序级结构被 runtime 看见，并转化为 KV cache 复用和约束解码加速？

# Method

前端语言提供 `gen`、`select`、`fork`、`join` 等原语，表达多调用、多分支和结构化生成。后端 runtime 维护 RadixAttention：用 radix tree 管理共享 prefix 的 KV cache，使不同请求或不同程序分支复用相同前缀。

调度器采用 cache-aware 策略，提高 prefix cache 命中率。对于结构化输出，SGLang 使用 compressed FSM 和 jump-forward，减少逐 token 约束检查的成本。系统还支持 API speculative execution，提前发起可能的模型调用。

# Dataset

实验覆盖 agent control、logical reasoning、few-shot learning、JSON decoding、RAG、多轮对话和多模态任务；模型包括 Llama 系列、Mixtral 和 LLaVA 等，在 A10G/A100 等 GPU 上评测。

# Main Findings

SGLang 在多类结构化 LLM 程序上相比 Guidance、vLLM、LMQL 等系统可获得最高约 6.4x 吞吐提升。RadixAttention 对共享 prefix、多分支和多轮对话场景尤其有效。

Compressed FSM 对 JSON 等受限格式生成有明显帮助，因为它减少了逐 token 的约束状态推进开销。

# Contribution

- 把 LLM 应用从“单次生成请求”抽象为可优化的语言模型程序。
- 提出 RadixAttention，用 radix tree 组织和复用 KV cache。
- 将 cache-aware scheduling、structured decoding 和 speculative execution 集成到统一 runtime。

# Limitations

Radix cache 管理会带来额外复杂度，且命中率依赖 workload 是否存在共享 prefix。结构化解码的 FSM 压缩也可能受 tokenizer、格式约束和概率分布偏移影响。

当前编译与调度对高度数据依赖的复杂控制流仍有局限，需要应用编写方式配合 runtime。

# Relation To My Research

SGLang 是从 serving system 走向 LLM program runtime 的代表。它与 PagedAttention 的 KV 管理、Orca 的 continuous batching、DistServe/Splitwise 的阶段拆分互补，适合放在同一个 LLM serving 系统专题下理解。

# Useful Quotes

- §1 可引用观点：真实 LLM 应用往往包含多次模型调用和共享上下文，单请求 serving 接口难以暴露这些优化机会。
- §3 可引用观点：RadixAttention 用树结构保存共享 prefix 的 KV cache，从而复用不同请求之间的公共上下文。
- §4 可引用观点：结构化输出可以通过 FSM 约束解码，但直接逐 token 推进会产生额外开销。

# Critical Notes

- SGLang 的核心不是某个单独 kernel，而是让“程序结构”进入 runtime 优化视野。
- 它与 vLLM 的关系很紧：vLLM 管理动态 KV blocks，SGLang 进一步管理 prefix 共享和程序分支。
- 如果研究 agent 或 RAG 系统，这篇比单纯 throughput serving 更贴近真实应用形态。
