# Zotero 同步入口

本目录用于保存 Zotero / Better BibTeX 导出的文献数据。

当前项目标准入口：

- `references.bib`

当前 `references.bib` 指向：

- `01_ToRead.bib`

建议在 Zotero Better BibTeX 中将目标 Collection 设置为自动导出到 `01_ToRead.bib`，并保持 `Keep updated` 开启。Codex 后续读取 `references.bib`，避免因 Collection 文件名变化而修改分析流程。

注意：

- Zotero 仍然是唯一可信文献库。
- 不要手动编辑 `.bib` 中的作者、年份、DOI、citation key 等文献信息。
- 如果需要修正文献元数据，请先在 Zotero 中修改，再让 Better BibTeX 自动更新导出文件。
