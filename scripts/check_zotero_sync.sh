#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIB_FILE="$ROOT_DIR/data/zotero/references.bib"

if [[ ! -e "$BIB_FILE" ]]; then
  echo "未找到 Zotero 同步入口: $BIB_FILE"
  exit 1
fi

ENTRY_COUNT="$(grep -c '^@' "$BIB_FILE" || true)"
FILE_COUNT="$(grep -c '^[[:space:]]*file[[:space:]]*=' "$BIB_FILE" || true)"
UPDATED_AT="$(stat -f '%Sm' "$BIB_FILE")"

echo "Zotero 同步入口: $BIB_FILE"
echo "文献条目数量: $ENTRY_COUNT"
echo "包含 PDF 路径的条目数量: $FILE_COUNT"
echo "文件更新时间: $UPDATED_AT"

if [[ "$ENTRY_COUNT" -eq 0 ]]; then
  echo "未检测到文献条目，请检查 Better BibTeX 导出设置。"
  exit 1
fi

echo "检查通过。"
