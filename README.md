# 科研文献分析工作区

## 项目用途

本工作区用于配合 Zotero 进行科研文献阅读、证据整理、综述写作和 Presentation 生成。它将 Zotero 中可信的文献资料转化为可追溯的阅读卡片、证据矩阵、综述草稿和演示文稿大纲。

## Zotero 与 Codex 的分工

- Zotero：作为唯一可信文献库，负责保存文献条目、PDF、元数据、citation key、标签、馆藏集合和阅读标注。
- Codex：负责基于 Zotero 导出的文献数据、PDF 和标注进行整理、分析、归纳、写作和生成结构化输出。
- 用户：负责确认研究问题、筛选标准、关键文献范围，以及最终学术判断。

## 工作流程说明

1. 在 Zotero 中建立或维护目标 Collection。
2. 从 Zotero 导出文献元数据、PDF 或标注到 `data/` 目录。
3. 为每篇核心文献建立 paper card，保存到 `notes/paper_cards/`。
4. 将文献证据整理进 `outputs/evidence_matrix.csv`。
5. 在 `notes/synthesis_notes/` 中形成主题归纳、争议点、研究空白和理论线索。
6. 生成 `outputs/literature_map.md`，梳理文献之间的关系。
7. 生成 `outputs/literature_review.md`，形成可追溯引用的综述草稿。
8. 生成 `outputs/slides_outline.md`，再进一步制作 Presentation 文件。

## 目录说明

- `AGENTS.md`：Codex 在本工作区内处理文献任务时必须遵守的协作规则。
- `research_question.md`：研究主题、研究问题、关键词和纳入排除标准。
- `data/zotero/`：保存 Zotero 导出的文献元数据、BibTeX、CSL JSON 或 RDF 文件。
- `data/pdfs/`：保存从 Zotero 或其他合法来源导出的原始 PDF。
- `data/annotations/`：保存 Zotero 标注、摘录、批注或阅读记录。
- `notes/paper_cards/`：保存单篇文献阅读卡片。
- `notes/synthesis_notes/`：保存跨文献主题归纳、比较分析和综述素材。
- `outputs/evidence_matrix.csv`：保存结构化文献证据矩阵。
- `outputs/literature_map.md`：保存文献脉络图、主题地图或研究谱系。
- `outputs/literature_review.md`：保存综述草稿。
- `outputs/slides_outline.md`：保存 Presentation 大纲。
- `outputs/presentation/`：保存最终演示文稿及相关素材。
- `scripts/`：保存未来自动化脚本和任务说明。
