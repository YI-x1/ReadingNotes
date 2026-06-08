# Citation Key

`yeOnlineSoftmaxFlashAttention`

文献：From Online Softmax to FlashAttention

# One Sentence Summary

这篇技术笔记从 safe softmax 与 online softmax 出发，逐步推导出 FlashAttention 如何在分块扫描 `K/V` 时维护 softmax 统计量并在线更新 attention 输出。

# Research Question

FlashAttention 的公式和 kernel 容易显得复杂。笔记要回答的是：能否从最简单的 softmax normalizer 递推开始，推导出 tile-wise attention 输出更新规则，从而理解 FlashAttention 为什么精确且省显存？

# Method

笔记先写出 self-attention：`O = softmax(QK^T)V`。标准实现会物化 `X = QK^T` 和 `A = softmax(X)`，而 safe softmax 需要先减去行最大值。

随后引入 online softmax：扫描每个 logit 时维护运行最大值 `m_i` 与归一化分母 `d_i`。当新的最大值出现时，旧分母按 `exp(m_old - m_new)` 缩放。

最后把标量 softmax 扩展到 attention 输出：不仅更新 `m` 和 `d`，还同步重缩放旧输出并加入当前 value block 的贡献。这就是 FlashAttention tile-wise 精确性的直观来源。

# Dataset

这是一篇解释性笔记，不做独立数据集评测。它主要服务于理解 Milakov 的 online normalizer 与 Dao 等人的 FlashAttention。

# Main Findings

笔记清楚说明了 FlashAttention 不是把 attention 近似掉，而是改变计算顺序。只要每次 tile 更新时正确维护 `m`、`d` 和输出向量，分块计算结果与整行 softmax 后乘 `V` 等价。

它也强调了 FlashAttention 的关键不只是 tiling，而是 tiling 与 online softmax 的结合；没有在线重缩放，跨 tile 的 softmax 无法保持全局归一化。

# Contribution

- 用较低门槛推导 online softmax 到 FlashAttention 的过渡。
- 把标量 softmax normalizer 与向量 attention output 的更新联系起来。
- 为阅读 CUDA kernel 中的 `m/l/O` 更新逻辑提供数学直觉。

# Limitations

作为技术笔记，它不提供完整工程实现、性能评估或边界条件讨论。实际 FlashAttention 还涉及 shared memory 布局、warp 级并行、dropout、mask、反向传播和不同 head dimension 的专门优化。

# Relation To My Research

这篇适合作为 FlashAttention 专题的“推导入口”。如果要从论文读到 CUDA 实现，建议先理解这篇里的输出递推，再看 Dao 论文和 kernel 中每个 block 如何更新 row-wise statistics。

# Useful Quotes

- Softmax 部分可引用观点：online softmax 的关键是当最大值更新时，把旧分母重缩放到新的数值参考系。
- Attention 部分可引用观点：attention 输出也必须随分母和最大值一起重缩放，才能跨 tile 合并。
- FlashAttention 部分可引用观点：FlashAttention 的省显存来自不物化 logits 和 attention scores，而不是牺牲精确性。

# Critical Notes

- 这篇笔记最适合补直觉，但不能替代原论文的 IO 分析和 kernel 工程细节。
- 推导中每个变量的“旧参考系到新参考系”转换，是理解实现时避免 bug 的关键。
- 可作为之后写 CUDA 注释或讲解图的公式来源。
