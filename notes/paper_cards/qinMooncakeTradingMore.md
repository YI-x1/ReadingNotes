---
status: draft
---

# Mooncake: Trading More Storage for Less Computation - A KVCache-centric Architecture for Serving LLM Chatbot

`qinMooncakeTradingMore`

文献：Mooncake: Trading More Storage for Less Computation - A KVCache-centric Architecture for Serving LLM Chatbot

元数据说明：Zotero 条目 `data/zotero/01_ToRead.bib` 未给出 year、venue、DOI 字段；PDF 封面显示该论文收录于 FAST 2025，并给出 USENIX 页面。以下分析仅基于 Zotero 条目、PDF 正文和摘要信息。

## 核心问题

Mooncake 面向 Kimi 这类真实长上下文聊天服务，问题不是单纯“prefill 和 decode 要不要分离”，而是：**能否把分布在 GPU 集群中的 CPU、DRAM、SSD、NIC 与 GPU 显存一起组织成全局 KVCache 系统，用更多存储和传输换取更少重复 prefill 计算，并在 TTFT/TBT SLO 下提升有效请求容量？**

## 方法概括

Mooncake 采用 KVCache-centric disaggregated architecture。它既分离 prefill cluster 和 decoding cluster，也通过 Mooncake Store 管理分布式 KVCache pool。全局调度器 Conductor 会根据请求的 prefix cache 命中情况、prefill 队列、KVCache 分布和 decoding 侧 TBT 约束，为请求选择 prefill instance 与 decoding instance。

请求大致经历四步：

1. 将可复用 prefix KVCache 加载到选中的 prefill instance。
2. 对未命中部分做 incremental prefill。
3. 将新增 KVCache 传给 decoding instance。
4. decoding instance 使用完整 KVCache 进入 continuous batching 并生成输出。

## 关键机制

- **全局 KVCache pool**：KVCache 被切成 paged blocks，存放在分布式 cache pool 中；block key 同时由自身 hash 和 prefix 决定，用于去重。
- **KVCache-aware prefill scheduling**：Conductor 不只按请求数做负载均衡，而是估计 prefix hit length、prefill 执行时间、KVCache transfer time 和队列等待时间，选择预计 TTFT 最低的 prefill instance。
- **hotspot migration / replication**：热点 KVCache block 会被复制到多个节点，缓解单点 cache 访问拥塞；冷块则可被换出。
- **Mooncake Store transfer engine**：提供 object-style `put/get/change_replica` 和 batch transfer API，支持 DRAM/VRAM 传输、GPU Direct RDMA、拓扑感知路径选择、多 NIC 分片、endpoint pooling 和失败重试。
- **chunked pipeline parallelism for prefill**：针对长上下文 prefill，Mooncake 用 chunked pipeline parallelism 避免跨节点 tensor parallelism 的频繁 all-reduce，同时减少与 KVCache transfer 的网络竞争。

## 主要证据

- 摘要与正文报告：在真实 traces 上，相比 baseline，Mooncake 在满足 SLO 的同时提升 effective request capacity 约 59% 到 498%。
- 生产部署报告：Mooncake 已在数千节点上运行，每天处理超过 100 billion tokens；在 A800 和 H800 集群中，相比此前系统分别多处理 115% 和 107% 请求。
- 全局 cache 与 local cache 对比：论文报告全局 cache hit rate 最高达到 local cache 的 2.36x，并最多节省 48% prefill computation time。
- transfer engine 对比：在 inter-node cache transfer 中，Mooncake transfer engine 相比 TCP 和 Gloo 更快；正文报告在 4x200 Gbps 和 8x400 Gbps 网络下，40 GB 传输场景可达到约 87 GB/s 和 190 GB/s。
- 带宽敏感性：论文指出当通信带宽超过 100 Gbps 时平均 TTFT 明显低于 recomputation baseline；低于 100 Gbps 时性能会显著下降。

## 优势

Mooncake 的最大优势是把 KVCache 从“某个请求在某台 GPU 上的临时状态”提升为“跨会话、跨请求、跨节点的全局资源”。这使它不仅能缓解 prefill/decode 干扰，还能直接减少重复 prefill 计算，尤其适合长上下文、多轮对话、工具/agent 固定系统提示词等 prefix reuse 明显的工作负载。

它的工程完整度也很高。论文不仅提出架构，还讨论了 transfer engine、RDMA 多 NIC、拓扑路径、endpoint pooling、故障处理、cache replication、prefill 调度和真实部署数据。相比只停留在集群配置或仿真层面的设计，它更接近生产 serving platform。

## 局限

- 收益高度依赖 prefix reuse、长上下文比例和全局 cache 命中率；对于短 prompt、低复用、短输出的普通请求，额外的全局 cache 管理和传输未必划算。
- 强依赖高速 RDMA 网络。论文自己的带宽实验显示，低于约 100 Gbps 时 TTFT 会明显恶化。
- 系统复杂度很高：需要维护全局 metadata、block key、replica、LRU eviction、transfer status、prefill/decode 配比和故障恢复。
- 相比 DistServe，Mooncake 对“不同 parallelism strategy 如何系统搜索最优 per-GPU goodput”的展开较少；相比 Splitwise，对异构硬件成本/功耗的建模不是主线。

## 与 Splitwise / DistServe 的关系

Mooncake 可以看作在 Splitwise 和 DistServe 的 prefill/decode disaggregation 基础上，把核心优化对象进一步推进到 **KVCache 生命周期**。Splitwise 强调阶段硬件差异和成本/功耗，DistServe 强调 TTFT/TPOT SLO 下的 goodput 和 placement，Mooncake 则强调用全局 KVCache 复用、迁移和调度减少重复 prefill，尤其面向真实长上下文聊天服务。

因此，Mooncake 不是简单替代 Splitwise 或 DistServe，而是把“阶段拆分”扩展成“全局 KVCache 资源编排”。如果系统负载具有高 prefix reuse 和长上下文特征，Mooncake 的处理方式更具工程纵深；如果负载没有明显复用，DistServe/Splitwise 的阶段级优化可能更直接。
