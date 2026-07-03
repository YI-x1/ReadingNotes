# Structured Generation / Constrained Decoding 读书 Q&A

本文是 `analysis.md` 的补充读书笔记，围绕 `gengGrammarConstrainedDecodingStructured2024`、`dongXGrammarFlexibleEfficient2025` 和 `gengJSONSchemaBenchRigorousBenchmark2025` 中容易混淆的概念做通俗解释。

## 1. 结构化 NLP 任务与形式语法

### Q1：什么叫“许多结构化 NLP 任务的输出空间可以用形式语法描述”？

很多 NLP 任务不是让模型随便写一段话，而是要求模型输出一种有固定格式的答案。例如：

```text
实体识别：输出每个词对应的标签
关系抽取：输出 (主体, 关系, 客体)
句法分析：输出一棵括号嵌套的句法树
JSON 生成：输出符合 schema 的 JSON
```

这些答案虽然具体内容不同，但“长什么样”通常有规则。形式语法可以理解为一套严格的格式说明：哪里能放括号，哪里必须放标签，哪里只能从候选集合里选，哪里不能提前结束。

所以 `gengGrammarConstrainedDecodingStructured2024` 的核心意思是：如果任务答案的合法形式可以写成语法规则，那么模型生成时就可以用这些规则过滤非法 token，而不一定要为每个任务单独 finetune 一个模型。

### Q2：为什么不直接让模型学会格式，而要在 decoding 时过滤非法 token？

因为模型即使理解任务，也可能生成格式错误的文本。例如少一个括号、多一个逗号、输出 schema 外的关系名，都会导致下游程序解析失败。

Constrained decoding 的思路是：模型每一步仍然预测下一个 token，但在真正选择 token 前，系统先屏蔽所有会违反语法的候选 token。这样模型还是原来的模型，只是输出空间被约束住了。

一个直观类比是：不是重新训练一个人如何填写表格，而是给他一个智能输入框。日期栏不能输入字母，JSON 里括号不能乱闭合，候选题只能选给定选项。

### Q3：CFG 是什么？

`CFG` 的全称是 **Context-Free Grammar**，中文通常译为**上下文无关文法**。

它是一套描述合法结构如何生成的规则。这里的“上下文无关”指的是：某个非终结符如何展开，主要由它自己决定，不依赖它左右两边的具体内容。

例如：

```text
句子 ::= 名词短语 动词短语
名词短语 ::= 限定词 名词
动词短语 ::= 动词
```

CFG 很适合描述括号嵌套、JSON 结构、句法树等结构化输出。

## 2. Input-Dependent Grammars

### Q4：什么是 input-dependent grammars？

`input-dependent grammars` 指的是：抽象输出格式可能固定，但每条输入对应的具体合法输出集合不同，所以要根据当前输入动态生成语法。

它不是说任务模板一定变了，而是说“这次允许填哪些具体内容”会随输入变化。

### Q5：entity disambiguation 里为什么需要 input-dependent grammar？

Entity disambiguation，即实体消歧，是从候选实体中选出当前文本里提到的真实实体。

比如输入 A 中的 `Apple`，候选可能是：

```text
Apple Inc.
Apple fruit
Apple Records
```

输入 B 中的 `Jordan`，候选可能是：

```text
Michael Jordan
Jordan, the country
Jordan River
```

抽象任务都是“从候选实体中选择正确实体”，但候选集合每条输入都不同。为了在 decoding 时禁止模型输出候选集合外的实体，grammar 必须根据当前输入临时生成。

所以动态变化的不是“选择一个实体”这个任务形状，而是“这次可选的实体字符串集合”。

### Q6：constituency parsing 里为什么需要 input-dependent grammar？

Constituency parsing，即成分句法分析，是把一句话解析成短语结构树。例如：

```text
The dog barked.
```

可以表示成：

```text
(S
  (NP (Det The) (N dog))
  (VP (V barked)))
```

这棵树的叶节点必须是原句里的 token：

```text
The
dog
barked
```

如果输入换成：

```text
A cat slept.
```

树的叶节点就必须换成：

```text
A
cat
slept
```

固定 grammar 可以规定括号如何配对、`S` 下面可以有 `NP` 和 `VP`，但它还要根据当前输入规定：叶节点必须复现当前句子的 token，通常还要保持原顺序，不能凭空生成输入里没有的词。

因此 constituency parsing 的抽象结构规则可以固定，但具体叶节点约束依赖输入。

### Q7：句法树到底是什么？

句法树可以理解为：把一句话拆成“哪些词先组成短语，这些短语再组成什么更大结构”的树。

例如：

```text
The dog barked.
```

可以拆成：

```text
        S
      /   \
    NP     VP
   /  \     |
 Det   N    V
 |     |    |
The   dog barked
```

这里：

```text
S   = Sentence，句子
NP  = Noun Phrase，名词短语
VP  = Verb Phrase，动词短语
Det = Determiner，限定词
N   = Noun，名词
V   = Verb，动词
```

句法树重点不是解释句子含义，而是解释语法结构：哪些词组成一个短语，哪些短语再组成句子。

### Q8：closed information extraction 是什么任务？

`closed information extraction` 可以译成**封闭式信息抽取**。

它要求从文本中抽取结构化信息，但抽取结果必须落在预先定义好的 schema、实体类型、关系类型或槽位集合里。

例如输入：

```text
乔布斯创立了苹果公司。
```

输出可能是：

```text
(乔布斯, 创立者-公司, 苹果公司)
```

这里的关系类型不是自由生成的，而是从固定 schema 中选择。如果 schema 里没有某个关系，模型就不应该输出它。

与它相对的是 open information extraction。Open IE 更自由，可能直接抽出：

```text
(乔布斯, 创立了, 苹果公司)
```

Closed IE 适合 grammar-constrained decoding，因为它的合法关系、槽位和输出格式更容易写成规则。

## 3. XGrammar 的 Mask Generation 优化

### Q9：XGrammar 和 Guidance 算什么？

XGrammar 和 Guidance 都不是模型本体，而是结构化生成的外接控制层。

可以分三层看：

```text
模型权重：
  Qwen、Llama、Mistral
  负责根据上下文算 logits

推理引擎 / generation runtime：
  vLLM、SGLang、llama.cpp、Transformers
  负责 prefill、decode、KV cache、sampling、batching

约束生成层：
  XGrammar、Guidance、Outlines
  负责根据 grammar / schema 生成合法 token mask 或控制生成路径
```

XGrammar 更像一个高性能 grammar-constrained decoding engine，重点是快速生成 token mask。Guidance 更像一个结构化生成框架，允许开发者用模板、选择、regex、JSON Schema 或 grammar 控制模型输出。

### Q10：它们怎样加入模型生成过程？

LLM 每一步生成大致是：

```text
已有前缀
  ↓
模型 forward，得到下一个 token 的 logits
  ↓
处理 temperature / top-p / top-k 等采样参数
  ↓
采样或选择下一个 token
  ↓
追加到输出
```

Grammar mask 插入的位置是：

```text
模型得到 logits 之后，真正采样之前
```

也就是：

```text
已有前缀
  ↓
LLM 计算下一个 token 的 logits
  ↓
XGrammar / Guidance 根据 grammar 或 schema 生成合法 token mask
  ↓
非法 token 的 logits 被屏蔽
  ↓
从剩下的 token 中采样或选择
  ↓
更新输出前缀和语法状态
```

所以结构化输出并不是模型不输出文本了。JSON、XML、句法树本质上也都是文本，只是这些文本必须符合严格结构。

### Q11：为什么单独一个模型权重不好直接接 grammar-mask？

因为 grammar-mask 需要接入“logits 后、采样前”的 generation loop。

一个模型权重文件，例如 Qwen 或 Llama，只负责把输入上下文映射成 logits。它本身不提供完整的解码控制流程。要使用 grammar-mask，需要加载它的推理框架支持这种约束逻辑。

所以不是：

```text
Qwen 自带 XGrammar
```

而是：

```text
Qwen 被 vLLM / SGLang / llama.cpp / Transformers 加载
推理框架掌控 generation loop
XGrammar / Guidance 在采样前接入约束
```

商业 API 的 structured outputs 看起来只是传一个 JSON Schema，但服务端内部也必须在某个层面处理结构约束，可能是 grammar-constrained decoding、parser、repair、重试或多种机制的组合。

### Q12：XGrammar 为什么要区分 context-independent tokens 和 context-dependent tokens？

朴素 constrained decoding 的瓶颈在 mask generation。

每一步模型都会对整个固定词表输出 logits。词表可能有几十万 token。语法约束系统要判断哪些候选 token 当前合法。如果每一步都对整个词表逐个运行完整 parser 检查，会很慢。

XGrammar 的核心思想是把判断拆成两层：

```text
第一层：根据当前 grammar stack state 判断现在期待什么
第二层：判断词表里的候选 token 能不能满足这些期待
```

第二层里有些判断可以提前缓存，有些必须运行时解释。于是 XGrammar 区分：

```text
context-independent tokens：
  对某个局部语法片段来说，token 是否匹配可以提前算

context-dependent tokens：
  token 是否匹配还要结合当前运行时细状态，必须现场解释
```

### Q13：context-independent 不是“不看状态也永远合法”吗？

不是。

`context-independent` 不是说某个 token 在任何状态下都合法，而是说：**一旦当前语法状态已经确定了某个局部期待，token 是否匹配这个局部期待可以提前知道**。

例如：

```text
当前期待：comma
token "," -> 合法
token "1" -> 不合法
```

这里 `1` 作为数字当然可能在别的状态合法，但在“当前期待 comma”这个局部片段下，它不合法。

所以 cache 存的不是：

```text
token "1" 永远合法
```

而是：

```text
期待 digit 时，"1" 合法
期待 comma 时，"1" 不合法
期待字段名 "name" 时，"1" 不合法
```

### Q14：局部匹配是怎么预计算的？

预计算的对象不是模型已经生成的 token，而是 tokenizer 固定词表里的所有候选 token。

对一个给定 LLM 来说，tokenizer 词表是固定的。例如：

```text
token id 123 -> "the"
token id 456 -> "{"
token id 789 -> "\"name\""
token id 1001 -> "1"
```

预计算时，系统拿某个局部语法片段和整个词表逐个匹配。

例如局部期待是字段名 `"name"`：

```text
token "\"name\"" -> 可以匹配
token "\"na"     -> 可以作为前缀匹配
token "name"     -> 不可以，因为少了引号
token "\"age\""  -> 不可以
```

于是缓存成：

```text
cache[field "name"] = 长度为 vocab_size 的 0/1 mask
```

真正 decoding 时，如果当前 grammar state 说“现在期待字段名 `"name"`”，就直接取这张 mask，而不是重新遍历整个词表。

### Q15：既然第二层第一次也要扫词表，优化在哪里？

优化不在于第一次凭空变快，而在于把重复的 token 匹配从 decode 热路径里搬出去，并让结果复用。

不分层的朴素做法是：

```text
每一步 decode：
  对词表中每个候选 token：
    从当前完整 grammar state 出发
    尝试消费这个 token
    能走通 -> mask[token] = 1
    走不通 -> mask[token] = 0
```

如果词表有 100k token，每一步都可能做 100k 次完整状态模拟。

分层做法是：

```text
预计算或首次遇到某个局部期待时：
  扫一遍固定词表
  建立 cache[局部期待]

每一步 decode：
  第一层根据 stack state 判断当前允许哪些局部期待
  第二层直接取对应 cache
  必要时对 context-dependent 部分运行时解释
```

例如 JSON 数组里会反复遇到 comma：

```json
[1, 2, 3, 4, 5]
```

不分层会每次期待 comma 都重新扫词表。分层只需要建立一次 `cache[comma]`，之后直接复用。

因此核心区别是：

```text
不分层：每步对每个候选 token 重新跑完整合法性检查
分层：当前状态 -> 局部期待；局部期待 -> 查缓存 mask；少数复杂情况再动态解释
```

### Q16：context-dependent 的例子是什么？

JSON string 内部的 escape 状态就是典型例子。

假设 token 是：

```text
abc"
```

如果当前在普通字符串内容中：

```json
"hello 
```

这个 token 可能合法，因为它可以继续生成 `abc` 并关闭字符串：

```json
"hello abc"
```

但如果当前刚生成了反斜杠：

```json
"hello \
```

接下来必须形成合法 escape，例如 `\"`、`\\`、`\n` 或 `\u1234`。这时同一个 token `abc"` 可能不合法，因为 `\a` 不是合法 JSON escape。

这类判断不能只靠“局部语法片段 + token 字符串”提前决定，还必须知道当前运行时已经走到哪个细状态。因此属于 context-dependent，需要 runtime 解释。

### Q17：不分层和分层发生在同一条 decode 时间线上吗？

是的，两者都发生在：

```text
logits 之后，采样之前
```

区别不是时间点，而是 mask generation 的内部实现。

不分层：

```text
LLM logits
  ↓
对每个 token：
  用当前完整 grammar state 试跑 token
  得到 mask
  ↓
采样
```

分层：

```text
LLM logits
  ↓
先用当前 grammar state 找到现在期待哪些局部片段
再查这些局部片段对应的预计算 mask
必要时处理 context-dependent 部分
得到 mask
  ↓
采样
```

### Q18：每次预测时词表会变吗？

不会。

对同一个模型和 tokenizer 来说，词表大小和 token id 集合是固定的。每一步 decode 面对的是同一套候选 token，变化的是：

```text
每个 token 的 logits 分数
当前 grammar state
当前允许的 token mask
```

因此 cache miss 通常不是因为“来了一个新 token”，而是因为遇到了新的 grammar fragment 或新的局部期待。

例如之前缓存过：

```text
cache[comma]
cache[digit]
cache[field "name"]
```

后来遇到新字段：

```text
field "email"
```

系统可能需要新建：

```text
cache[field "email"]
```

但它仍然是在同一个固定 tokenizer 词表上计算 mask。

## 4. Guidance、XGrammar 与 JSONSchemaBench 的关系

### Q19：Guidance 是什么？

Guidance 是一个结构化生成 / 约束生成框架。它不是模型，而是包在模型外面的一层 generation framework。

它允许开发者用模板、选择、regex、JSON Schema、grammar 等方式控制模型输出。例如要求某个位置必须从 `yes/no` 中选择，或者整个输出必须符合 JSON Schema。

在 `gengJSONSchemaBenchRigorousBenchmark2025` 的语境中，Guidance 是被评测的 structured output framework 之一。JSONSchemaBench 报告 Guidance 在多项覆盖与综合评测上表现较好，说明它对真实 JSON Schema 的支持相对完整。

### Q20：XGrammar 是 2025 年的工作，为什么在 JSONSchemaBench 里不一定比 Guidance 好？

XGrammar 是 `dongXGrammarFlexibleEfficient2025`，主攻的是高效执行 CFG / JSON Schema 约束，尤其是降低 mask generation latency，让 constrained decoding 可以进入高吞吐 serving。

Guidance 在 JSONSchemaBench 中表现好，更多是因为它对真实 JSON Schema 的覆盖和语义处理更完整。真实 JSON Schema 不只是括号、字段名和类型，还包括：

```text
anyOf / oneOf / allOf
additionalProperties
patternProperties
required
enum / const
嵌套 schema
数组长度
字符串 pattern
数字范围
引用和组合约束
```

所以“快”和“覆盖全”是两个不同维度。XGrammar 强在执行效率，Guidance 在 JSONSchemaBench 里强在 schema 覆盖和综合可靠性。JSONSchemaBench 的价值正是提醒我们：structured outputs 不能只看能否保证合法 JSON，还要看效率、覆盖和任务质量。

## 5. 一句话总结

结构化生成的基本图景是：

```text
LLM 仍然是下一个 token 预测器；
形式语法 / JSON Schema 描述什么输出合法；
XGrammar / Guidance 在 logits 后、采样前约束候选 token；
JSONSchemaBench 用真实 schema 检验这些系统是否高效、覆盖充分且不伤害任务质量。
```
