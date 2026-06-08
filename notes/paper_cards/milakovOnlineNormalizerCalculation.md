# Citation Key

`milakovOnlineNormalizerCalculation`

文献：Online Normalizer Calculation for Softmax

# One Sentence Summary

这篇论文提出在线计算 softmax 归一化项的方法，用运行中的最大值和分母递推替代传统 safe softmax 的多次内存遍历。

# Research Question

数值稳定 softmax 通常先找最大值，再计算指数和，最后归一化输出。论文要回答的是：能否在一次流式扫描中同时更新最大值和归一化分母，从而减少内存访问并提升 softmax 及 Softmax+TopK 性能？

# Method

对于输入序列，在线维护当前最大值 `m_j` 和归一化分母 `d_j`。当新元素 `x_j` 到来时，先更新 `m_j = max(m_{j-1}, x_j)`，再把旧分母按新最大值重新缩放，并加入当前元素的贡献。

这个递推还可以写成可并行归约的二元操作，使 GPU reduction 能在多个线程块中合并局部结果。论文进一步将 softmax 与 TopK 融合，减少中间概率写回。

# Dataset

论文以 GPU 上的 softmax 和 Softmax+TopK kernel 为主要评估对象，关注大 vocabulary 场景下的内存访问次数和 kernel 性能。

# Main Findings

在线 normalizer 能减少 safe softmax 的访存轮次。单独 softmax 可获得约 1.3 倍加速，Softmax+TopK 融合在部分场景可达到约 5 倍加速。

最重要的发现是：数值稳定性不必依赖“先完整求 max，再完整求 sum”的两阶段物化流程；只要递推中正确重缩放旧分母，就能保持等价结果。

# Contribution

- 给出稳定 softmax normalizer 的在线递推形式。
- 说明该递推可并行化，不只适用于串行流式计算。
- 展示 Softmax+TopK 融合时在线 normalizer 能显著减少中间数据流动。

# Limitations

如果需要输出完整 softmax 概率，最终仍需要再访问输入或缓存中间值来归一化每个元素；在线算法主要减少 normalizer 计算阶段的访存。

递推引入额外指数和缩放操作，收益依赖 softmax 是否 memory-bound 以及输入规模。

# Relation To My Research

这是理解 FlashAttention 的数学前置。FlashAttention 的 tile-wise attention 本质上依赖同类在线 softmax 统计量，把 softmax 从“整行矩阵操作”变成“可流式拼接的块级更新”。

# Useful Quotes

- §2 可引用观点：safe softmax 的稳定性来自减去最大值，但传统实现需要多次遍历输入。
- §3 可引用观点：当最大值变化时，旧分母可以通过指数因子缩放到新的参考系。
- §4 可引用观点：Softmax 与 TopK 融合时，避免写回完整概率分布能带来更大收益。

# Critical Notes

- 这篇论文的公式很短，但影响很大：它把 softmax 变成了可组合的 streaming primitive。
- FlashAttention 中最容易忽略的细节就是“旧 block 的输出也要随新的 max/sum 重新缩放”。
- 实现时要防止把在线 softmax误解成近似算法；它是精确等价的数值稳定重排。
