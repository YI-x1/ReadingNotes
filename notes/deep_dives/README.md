# Deep Dives

本目录用于保存重要论文的深度专题分析。

约定：

- 每篇重要论文使用一个以 `citation_key` 命名的子目录。
- 默认维护 `analysis.md` 和 `figures/`。
- 只有用户明确要求时，才额外生成 PDF 阅读版。
- `analysis.md` 用于长期迭代、补充公式推导、图示解释和个人理解。
- `figures/` 用于保存自绘图、论文图截图、公式示意图或其他辅助材料。
- deep dive 论文分析应图文并茂：优先从论文 PDF 中裁取关键原图，保留 figure 编号或 caption，并在 `analysis.md` 的相关段落就近引用；只有论文原图不足以解释时，才补充自绘图。

建议结构：

```text
deep_dives/
└── citation_key/
    ├── analysis.md
    └── figures/
```

专题合集：

- 当多篇论文强关联、需要按主题连续阅读时，可以额外建立一个以主题命名的合集目录。
- 合集目录同样维护 `analysis.md` 和 `figures/`，用于串联多篇论文的共同问题、推理链和图示。
- 单篇论文的 `citation_key/` 目录可以放在合集目录下，作为原始精讲和长期迭代入口。

当前专题：

- `onlineSoftmaxToFlashAttention/`：Online Softmax 到 FlashAttention。
- `llmServingSystems/`：Orca、PagedAttention/vLLM、SGLang 组成的 LLM serving systems 主题。
- `prefillDecodeDisaggregation/`：Splitwise、DistServe、Mooncake 组成的 prefill/decode 分离与全局 KVCache 编排主题。
- `structuredGenerationConstrainedDecoding/`：GCD、XGrammar、JSONSchemaBench 组成的结构化生成与约束解码主题。
