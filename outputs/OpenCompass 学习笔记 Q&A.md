# OpenCompass 学习笔记 Q&A

本文档整理自对 OpenCompass 评测框架的学习与讨论，覆盖配置结构、内置 benchmark、推理/评测模式、模型 wrapper 等主题。内容基于 [OpenCompass 官方文档](https://opencompass.readthedocs.io/) 与仓库 `dataset-index.yml`，供后续查阅。

---

## 1. OpenCompass 可以自定义哪几类东西？

OpenCompass 的评测流水线可概括为：

**配置 → 推理（Infer）→ 评测（Eval）→ 可视化（Viz）**

五类可定制项分别卡在流水线不同环节：

| 类别 | 解决什么问题 | 主要配置位置 | 影响阶段 |
|------|--------------|--------------|----------|
| 自定义数据集 | 内置 benchmark 覆盖不了业务场景 | `datasets` + `reader_cfg` | 数据准备 |
| 自定义 PromptTemplate | 同一道题不同问法分数差很多 | `infer_cfg.prompt_template` | Infer |
| 自定义模型 / wrapper | 不同推理后端接口不同 | `models[].type` + 参数 | Infer |
| 自定义 evaluator / LLM-as-Judge | 开放题无法用字符串精确匹配 | `eval_cfg.evaluator` | Eval |
| 流程控制 | 控制跑哪一段、是否复用中间结果 | `run.py -m` / `-r`；`n`/`k` 采样；`CascadeEvaluator` | 调度 |

**心智模型：**

```
分数 = Evaluator( ModelWrapper( PromptTemplate( Dataset样本 ) ) )
```

- 换 **Dataset** → 换考什么  
- 换 **PromptTemplate** → 换怎么考  
- 换 **ModelWrapper** → 换谁来考、怎么调  
- 换 **Evaluator** → 换怎么打分  
- 换 **流程参数** → 换跑几次、跑哪几步、是否复用推理结果  

### 流程相关的补充

**阶段控制（`run.py -m`）：**

| 模式 | 作用 |
|------|------|
| `-m all` | 推理 + 评测 + 可视化（默认） |
| `-m infer` | 只生成预测 |
| `-m eval` | 基于已有预测只跑评测 |
| `-m viz` | 只汇总结果为表格 |

配合 `-r` 可复用某次 timestamp 下的推理结果。

**重复采样：** 在 dataset 配置里设 `n`（每题跑几次）、`k`（如 G-Pass@k），用于数学、代码等随机性强的任务。

**级联评测（`CascadeEvaluator`）：** 判分阶段内部先规则后 LLM（`parallel=False`），或两者并行任一判对即对（`parallel=True`）。这是 evaluator 层面的编排，不是 `run.py` 的 `-m`。

---

## 2. `reader_cfg`、`infer_cfg`、`eval_cfg` 分别是什么？

用**老师出题、学生答题、老师改卷**来类比：

| 配置 | 通俗说法 | 一句话 |
|------|----------|--------|
| `reader_cfg` | 读题库 | 从文件里读出题目和标准答案，整理成 OpenCompass 能用的格式 |
| `infer_cfg` | 出题 + 让学生答 | 把题目包装成 prompt，交给模型生成答案 |
| `eval_cfg` | 改卷打分 | 拿模型答案和标准答案比，算出分数 |

### `reader_cfg`：读题库

告诉 OpenCompass 数据文件长什么样、读哪些列：

- **输入列**（给模型看）：如 `sentence1`, `question`
- **标准答案列**（改卷用）：如 `label`, `answer`

不调用模型，只做数据准备。

### `infer_cfg`：出题 + 让学生答

把 `reader_cfg` 读出的字段拼成模型看到的 prompt，再让模型生成回答。通常包含：

- **`prompt_template`**：prompt 长什么样（`{question}` 等占位符会替换）
- **`retriever`**：是否加 few-shot 例题（0-shot 用 `ZeroRetriever`）
- **`inferencer`**：怎么让模型回答（常见 `GenInferencer` 生成式）

### `eval_cfg`：改卷打分

模型答完后，与标准答案比较，算准确率等指标。可含：

- **`evaluator`**：如 `AccEvaluator`、`GenericLLMEvaluator`
- **`pred_postprocessor`**：从长输出里抽取最终答案

### 三者为何拆开？

三步可独立更换：

- 换数据 → 主要改 `reader_cfg`
- 换 prompt → 主要改 `infer_cfg`
- 换打分方式 → 主要改 `eval_cfg`，且常**不必重新跑推理**

```
原始 JSON → reader_cfg → infer_cfg → eval_cfg → 指标
```

---

## 3. OpenCompass 内置 benchmark 有哪些？

官方 `dataset-index.yml` 中约有 **189 个数据集**配置，位于仓库 `configs/datasets/`。完整可检索列表见：[Dataset Statistics](https://opencompass.readthedocs.io/zh-cn/stable/dataset_statistics.html)。

**本地查看：**

```bash
python tools/list_configs.py          # 列出所有配置
python tools/list_configs.py mmlu     # 模糊搜索
```

命令行评测用的是**配置名**（如 `mmlu_gen`、`gsm8k_gen`），不是论文简称。

### 按能力分类（常见代表）

| 类别 | 代表 benchmark |
|------|----------------|
| 综合知识 / 理解 | MMLU, MMLU-Pro, CMMLU, C-Eval, GPQA, SimpleQA |
| 数学 | GSM8K, MATH, AIME2024, MathBench, SVAMP |
| 推理 | BBH, HellaSwag, ARC, StrategyQA, HLE, AGIEval |
| 代码 | HumanEval, MBPP, LiveCodeBench, APPS, DS-1000 |
| 考试 | C-Eval, GAOKAOBench, RACE, AIME2024 |
| 长上下文 | LongBench, InfiniteBench, RULER, NeedleBench V2 |
| 中文 NLP | CLUE 系列, FewCLUE 系列, LCSTS |
| 安全 | TruthfulQA, CrowsPairs, CVALUES |
| 主观 / 对齐 | AlignBench, AlpacaEval, Arena-Hard, MT-Bench-101 |
| 工具 | T-Eval |
| 医学 / 专业 | MedQA, LawBench, ChemBench, FinanceIQ |

另有 `configs/datasets/collections/` 下的**预设组合**（如 `base_medium`）用于中等规模综合评测。

**实用建议：** 不要一次跑完全部；入门常见组合为 MMLU + C-Eval（综合）、BBH + GSM8K（推理）、HumanEval + MBPP（代码）。配置名以 `_gen.py` 结尾的通常为推荐配置。

---

## 4. 文件名里的 `_gen_db509b` 这类 hash 是什么？

命名格式：

```
{数据集名}_{评测方式}_{prompt版本hash}.py
```

示例 `CLUE_afqmc_gen_db509b.py`：

| 部分 | 含义 |
|------|------|
| `CLUE_afqmc` | 数据集 |
| `gen` | 生成式评测 |
| `db509b` | 该套 **prompt 配置** 的版本指纹 |

同一 benchmark 可有多种 prompt 策略；hash 用于区分版本、便于复现。由 `infer_cfg`（主要是 `prompt_template`）等内容导出，可用 `tools/update_dataset_suffix.py` 更新。

**`gen` vs `ppl`：**

| 后缀 | 含义 | 典型场景 |
|------|------|----------|
| `gen` | 生成式，模型续写答案 | Chat 模型、开放题、CoT |
| `ppl` | 判别式，用困惑度选选项 | Base 模型选择题 |

无 hash 的文件（如 `gsm8k_gen.py`）通常指向该评测方式下**当前推荐**的配置。

---

## 5. `infer_cfg` 具体怎么工作？（含 GSM8K 示例）

`infer_cfg` 管推理阶段：把 `reader_cfg` 的字段 → 拼 prompt → 调模型生成。

### GSM8K 简化示例

```python
gsm8k_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # few-shot 例题（写死在 template 里）
                dict(role='HUMAN', prompt="Question: ...\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt='...\nThe answer is 4\n'),
                # ... 更多例题 ...
                # 当前待测题
                dict(role='HUMAN', prompt="Question: {question}\nLet's think step by step\nAnswer:"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)
```

### 字段说明

**`prompt_template`**

- 占位符必须与 `reader_cfg` 列名一致（GSM8K 用 `{question}`，不是随意的 `{input}`）
- `output_column` 字段在拼 prompt 时会被**遮住**，防止泄题
- `role='HUMAN'` / `'BOT'`：对话格式；Chat 模型常用

**`retriever`**

| 类型 | 含义 |
|------|------|
| `ZeroRetriever` | 0-shot，不动态抽例题 |
| `FixKRetriever` 等 | 从数据中抽 few-shot |

注意：例题也可**写死在 `template.round` 里**（如 GSM8K），此时仍用 `ZeroRetriever`。

**`inferencer`**

| 类型 | 行为 |
|------|------|
| `GenInferencer` | 生成式续写 |
| `PPLInferencer` | 对候选算困惑度（配合 `ppl` 配置） |

---

## 6. Perplexity（困惑度）是什么？`ppl` 评测怎么用？

**困惑度（PPL）** 衡量模型看到一段文字时有多「意外」：

- PPL **低** → 文字对模型很「顺」、好预测  
- PPL **高** → 模型觉得别扭  

### 在 OpenCompass `ppl` 评测中

用于**选择题**：把每个选项拼成完整段落，分别算 PPL，**选 PPL 最低的选项**。

示例：法国的首都是？A. 柏林 B. 巴黎 C. 罗马

```
段落 B 的 PPL 最低 → 选 B
```

模型不需要生成「我选 B」，只需对给定文字打分——即**判别式**评测。

### `gen` vs `ppl`

| | **ppl** | **gen** |
|--|---------|---------|
| 模型做什么 | 对给定文字算困惑度 | 续写答案 |
| 怎么选题 | PPL 最低的选项 | 从生成文本里抽 A/B/C/D |
| 适用 | Base 模型、有 logprob 接口 | Chat / API 模型 |
| 后处理 | 简单 | 常需从长文本抽答案 |

经验：Base 模型选择题常用 `ppl`；Chat 模型几乎全用 `gen`；需要 CoT 过程时也用 `gen`。

---

## 7. 模型 wrapper 是什么？

**Wrapper = OpenCompass 与真实推理后端之间的适配器。**

评测流水线只认统一接口（`generate` / `get_ppl`），wrapper 把调用翻译成 HuggingFace、LMDeploy、API 等具体实现。

```
OpenCompass → wrapper (type) → 推理运行时 → 模型权重 (path)
```

### Wrapper 需实现的方法

| 方法 | 用途 | 对应评测 |
|------|------|----------|
| `generate()` | 给定 prompt 续写 | `gen` |
| `get_ppl()` | 给定文本算困惑度 | `ppl` |
| `get_token_len()` | 算 token 长度 | 截断、batch |

### 常见内置 wrapper

| Wrapper | 典型场景 |
|---------|----------|
| `HuggingFaceCausalLM` | 本地 HF 直载 |
| `TurboMindModel` / `TurboMindModelwithChatTemplate` | LMDeploy 加速 |
| `OpenAI` / `OpenAISDK` | API 模型 |
| 自定义 `BaseModel` 子类 | 自研推理引擎 |

配置示例：

```python
models = [
    dict(
        type=TurboMindModelwithChatTemplate,  # wrapper
        abbr='qwen-7b',
        path='Qwen/Qwen2.5-7B-Instruct',      # 模型
        engine_config=dict(tp=1),
        gen_config=dict(temperature=0.6),
        max_out_len=2048,
        run_cfg=dict(num_gpus=1),
    )
]
```

---

## 8. `type` 和 `path` 分别是什么？`type` 是供应商吗？

**不是。** `type` 不是模型厂商，而是**调用方式 / wrapper 类**；`path` 是**具体模型标识**。

| 字段 | 准确理解 |
|------|----------|
| `type` | 用哪种 wrapper / 哪条推理通路 |
| `path` | 哪个模型（HF ID、本地路径、API model name） |

同一 `path` 可配不同 `type`：

```python
# 同一 Qwen 模型，三种调用方式
dict(type=HuggingFaceCausalLM, path='Qwen/Qwen2.5-7B-Instruct')
dict(type=TurboMindModelwithChatTemplate, path='Qwen/Qwen2.5-7B-Instruct')
dict(type=OpenAISDK, path='Qwen/Qwen2.5-7B-Instruct', openai_api_base='...')
```

**心智模型：** `path` 选「评谁」；`type` 选「怎么跑」。

---

## 9. 不同 wrapper 有什么区别？同一模型能用不同 wrapper 吗？

### 差别主要在五方面

1. **推理运行时**（Transformers / TurboMind / vLLM / HTTP）  
2. **性能**（吞吐、延迟、显存）  
3. **能力**（是否支持 `gen` / `ppl`）  
4. **接入形态**（本地权重 vs 远程 API）  
5. **格式处理**（`meta_template`、后处理）

### 同一模型通常可用多种 wrapper

只要权重/API 形态支持。例如 Qwen 可用 HF、LMDeploy、或 vLLM 起的 API 服务。

### 何时必须用特定 wrapper？

- 仅 API、无本地权重（如 GPT-4）→ 只能 API wrapper  
- 需要 `ppl` 评测 → 需实现 `get_ppl()`（API 通常不支持）  
- 引擎只支持部分架构 → TurboMind/vLLM 有支持列表  
- 对比推理引擎本身 → 每个引擎需对应 wrapper  
- Chat 模型 → 常需 `meta_template` 或带 ChatTemplate 的 wrapper  

### 换 wrapper 会影响分数吗？

理论上不应差很多，实际可能因精度、采样参数、chat template、截断长度等略有偏差。**比模型能力时统一 wrapper 和生成参数；比推理引擎时固定模型与 benchmark、换 wrapper。**

---

## 10. 三层架构：wrapper、`type`/`path` 与 engine 的关系

容易混淆的是：配置里只有 `type` 和 `path`，但 wrapper **内部**还会调用真正的推理运行时（engine）。

```
OpenCompass 评测流水线
        │
        ▼  generate() / get_ppl()
   Wrapper（配置里的 type）
        │
        ▼
   推理运行时 / Engine（一般不单独作为 type 配置）
   Transformers / TurboMind / vLLM / HTTP
        │
        ▼
   模型权重（配置里的 path）
```

| 配置 `type` | 配置 `path` | wrapper 内部实际使用 |
|-------------|-------------|----------------------|
| `HuggingFaceCausalLM` | `Qwen/Qwen2.5-7B` | Transformers + PyTorch |
| `TurboMindModel` | `Qwen/Qwen2.5-7B` | LMDeploy TurboMind engine |
| `OpenAISDK` | `Qwen/Qwen2.5-7B` | HTTP；服务端可能是 vLLM 等 |

- **vLLM、TurboMind** 都是 **inference engine**  
- **`type`** 不是 engine，而是「某种 wrapper 绑定了某种 engine」  
- 表里说的「后端」指 wrapper **内部依赖**的运行时，不是与 `type`、`path` 并列的第三个标准配置项  

---

## 11. wrapper「是否支持 gen / ppl」是什么意思？

指 wrapper **能不能提供** OpenCompass 做两类评测所需的接口，与吞吐延迟无关。

| 模式 | 需要的方法 | 数据集侧 |
|------|------------|----------|
| **gen** | `generate()` | `GenInferencer`，`*_gen` 配置 |
| **ppl** | `get_ppl()` | `PPLInferencer`，`*_ppl` 配置 |

| Wrapper | gen | ppl | 原因 |
|---------|-----|-----|------|
| `HuggingFaceCausalLM` | ✅ | ✅ | 本地模型，logits 可得 |
| `TurboMindModel` | ✅ | 视版本 | 偏生成 |
| `OpenAI` / `OpenAISDK` | ✅ | ❌ | API 通常无 perplexity |

跑 `mmlu_ppl` 但 wrapper 不支持 `get_ppl()` 会无法评测；评 GPT-4 只能选 `*_gen` 配置。

---

## 12. LLM-as-Judge 的提问模板怎么设计？

Judge 模板写在 `eval_cfg` 的 `GenericLLMEvaluator` 里，给**裁判模型**用，与 `infer_cfg` 给被测模型的 prompt **无关**。

### 典型结构：SYSTEM + HUMAN

```python
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=JUDGE_TEMPLATE),
                ],
            ),
        ),
        judge_cfg=judge_model[0],
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
)
```

### 可用占位符

| 占位符 | 含义 |
|--------|------|
| `{problem}` / `{question}` | 原题 |
| `{answer}` | 标准答案 |
| `{prediction}` | 被测模型输出 |

### 官方示例模板

```python
JUDGE_TEMPLATE = """
Please evaluate whether the following response correctly answers the question.
Question: {problem}
Reference Answer: {answer}
Model Response: {prediction}

Is the model response correct? If correct, answer "A"; if incorrect, answer "B".
""".strip()
```

### 设计四要素

1. **角色**：SYSTEM 定义「你是评测助手」  
2. **判分标准**：语义等价 vs 严格匹配 vs 只看最终数值等  
3. **输入包装**：用标签分区展示 question / answer / prediction（长 CoT 时尤其重要）  
4. **输出格式**：必须与 `dict_postprocessor` 一致  

默认 `generic_llmjudge_postprocess` **只认 A / B**。改用 `CORRECT`/`INCORRECT` 等需自定义后处理器。

### 数据流

```
infer：被测模型 + infer_cfg → prediction
eval：problem + answer + prediction → 填入 JUDGE_TEMPLATE → 裁判模型 → A/B → 汇总 accuracy
```

裁判模型宜更强（如 Qwen2.5-32B-Instruct）、温度宜低（如 0.001）。

---

## 附录：配置字段速查

| 你想改… | 配置位置 |
|---------|----------|
| 题目从哪来、字段怎么读 | `reader_cfg` |
| 模型看到什么 prompt | `infer_cfg` |
| 用什么方式调模型 | `models[].type` + 参数 |
| 用什么模型 | `models[].path` |
| 答案怎么算分 | `eval_cfg.evaluator` |
| 跑全流程还是某一段 | `run.py -m` / `-r` |
| 每题采样几次 | `datasets[].n`, `k` |

---

## 参考链接

- [OpenCompass 文档](https://opencompass.readthedocs.io/)
- [配置数据集](https://doc.opencompass.org.cn/zh_CN/user_guides/datasets.html)
- [Prompt Template](https://opencompass.readthedocs.io/en/latest/prompt/prompt_template.html)
- [准备模型](https://doc.opencompass.org.cn/zh_CN/latest/user_guides/models.html)
- [LLM as Judge](https://opencompass.readthedocs.io/en/latest/advanced_guides/llm_judge.html)
- [数据集统计](https://opencompass.readthedocs.io/zh-cn/stable/dataset_statistics.html)
- [GitHub: open-compass/opencompass](https://github.com/open-compass/opencompass)
