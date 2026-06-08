# Citation Key

`kwonEfficientMemoryManagement2023`

文献：Efficient Memory Management for Large Language Model Serving with PagedAttention

# One Sentence Summary

PagedAttention 把 KV cache 管理类比为操作系统分页，用固定大小 KV block 和 block table 消除连续显存分配带来的碎片与浪费，从而提升 LLM serving 吞吐。

# Research Question

LLM 推理中 KV cache 会随输出长度增长，并且每个请求长度不可预知。论文要回答的是：如何在不要求连续显存的情况下高效存取 KV cache，并支持动态 batching、parallel sampling 和 beam search？

# Method

vLLM 将每个序列的 KV cache 切成固定大小 block，逻辑 token 位置通过 block table 映射到物理 KV block。PagedAttention kernel 在 attention 时根据 block table 读取非连续 KV cache。

系统层面提供 block 分配、释放、fork、append 和 copy-on-write。当多个候选序列共享 prefix 时，可以共享相同物理 KV block；只有分叉后写入新 token 时才复制。

# Dataset

实验覆盖 OPT-13B/66B/175B、LLaMA-13B 等模型，使用 ShareGPT、Alpaca、WMT16 翻译和聊天类 workload，在 A100 GPU 上与 FasterTransformer、Orca 类系统对比。

# Main Findings

vLLM 相比已有 serving 系统通常带来 2-4 倍吞吐提升。固定 block size 能显著降低显存碎片，prefix 共享在 parallel sampling 和 beam search 中尤其有效。

论文也显示 block table 间接寻址会让 attention kernel 有一定开销，但整体系统吞吐收益足以覆盖这部分成本。

# Contribution

- 将 OS paging 思想迁移到 LLM KV cache 管理。
- 提出 PagedAttention kernel，使非连续 KV cache 仍可被高效 attention 访问。
- 用 copy-on-write 支持多候选生成中的 prefix 共享。

# Limitations

PagedAttention 引入了 block table lookup 和非连续访存，kernel 本身可能比连续 KV cache 更慢。block size 选择存在折中：太小会增加元数据和查表开销，太大又会增加内部碎片。

该工作主要解决在线 serving 的 KV cache 管理，对 prefill/decode 分离、结构化程序执行等问题需要与其他系统组合。

# Relation To My Research

这是研究 LLM serving 时绕不开的基础论文。它把“显存容量”从静态模型权重问题扩展为动态请求生命周期问题，也为后续 SGLang 的 RadixAttention、prefix cache 和 serving scheduler 奠定了系统背景。

# Useful Quotes

- §1 可引用观点：KV cache 是 LLM serving 中随请求动态增长的主要显存压力来源。
- §3 可引用观点：通过 block table，逻辑连续的 KV cache 可以映射到物理非连续显存。
- §4 可引用观点：copy-on-write 让多个生成分支共享 prefix cache，减少重复存储。

# Critical Notes

- PagedAttention 的价值不只是节省显存，它还让调度器能更自由地接纳、暂停和恢复请求。
- 与 FlashAttention 不同，它优化的是跨请求/跨时间的 KV cache 管理，而不是单次 attention 的 SRAM/HBM tile IO。
- 读实现时要重点看 block table 数据结构、KV block layout、kernel 中如何把 token index 转成物理 block offset。
