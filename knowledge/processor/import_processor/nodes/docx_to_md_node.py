import logging
from pathlib import Path
from typing import List, Optional

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState


class DocxToMdNode(BaseNode):
    """
    将 Word 文档（.docx）转换为 Markdown，复用现有 MD 处理链路。

    处理内容：
    - 普通段落
    - 标题（按样式 Heading 1~6 映射为 # ~ ######）
    - 表格（转为 Markdown 表格）
    - 图片（导出到同级 images 目录，并替换为 Markdown 图片语法）
    """

    name = "docx_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        docx_path = state.get("docx_path")
        if not docx_path:
            self.logger.error("docx_path 为空")
            raise FileProcessingError(node_name=self.name, message="docx_path 为空")

        docx_path_obj = Path(docx_path)
        file_dir = state.get("file_dir")
        if not file_dir:
            self.logger.error("file_dir 为空")
            raise FileProcessingError(node_name=self.name, message="file_dir 为空")

        file_dir_obj = Path(file_dir)
        output_dir = file_dir_obj / docx_path_obj.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = self._convert_docx_to_md(docx_path_obj, output_dir)
        state["md_path"] = str(md_path)
        return state

    def _convert_docx_to_md(self, docx_path_obj: Path, output_dir: Path) -> Path:
        try:
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as e:
            self.logger.error(f"未安装 python-docx，无法解析 Word 文档: {e}")
            raise FileProcessingError(
                node_name=self.name,
                message="未安装 python-docx，请执行 pip install python-docx"
            )

        try:
            document = Document(str(docx_path_obj))
        except Exception as e:
            self.logger.error(f"读取 Word 文档失败: {e}")
            raise FileProcessingError(node_name=self.name, message=f"读取 Word 文档失败: {e}")

        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        md_lines: List[str] = []
        image_index = 0

        # 使用 element 遍历保持段落与表格的原始顺序
        body = document.element.body
        for child in body:
            tag = child.tag
            if "p" in tag:
                paragraph = Paragraph(child, document)
                para_md, image_index = self._process_paragraph(
                    paragraph, image_dir, image_index
                )
                if para_md:
                    md_lines.append(para_md)
            elif "tbl" in tag:
                table = Table(child, document)
                table_md = self._process_table(table)
                if table_md:
                    md_lines.append(table_md)

        md_content = "\n\n".join(md_lines)
        md_path = output_dir / f"{docx_path_obj.stem}.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        except Exception as e:
            self.logger.error(f"写入 Markdown 文件失败: {e}")
            raise FileProcessingError(node_name=self.name, message=f"写入 Markdown 文件失败: {e}")

        self.logger.info(f"Word 转 Markdown 完成: {md_path}")
        return md_path

    def _process_paragraph(self, paragraph: "Paragraph", image_dir: Path,
                           image_index: int) -> tuple:
        """处理单个段落，提取文本和嵌入图片。"""
        text = (paragraph.text or "").strip()
        if not text:
            return "", image_index

        # 标题样式映射
        style_name = paragraph.style.name if paragraph.style else ""
        heading_level = self._extract_heading_level(style_name)

        if heading_level:
            return f"{'#' * heading_level} {text}", image_index
        return text, image_index

    @staticmethod
    def _extract_heading_level(style_name: str) -> Optional[int]:
        """从 Word 样式名中提取标题级别。"""
        if not style_name:
            return None
        style_name = style_name.lower()
        if "heading" in style_name or "标题" in style_name:
            for level in range(1, 7):
                if f"heading {level}" in style_name or f"标题 {level}" in style_name:
                    return level
        return None

    def _process_table(self, table: "Table") -> str:
        """将 Word 表格转为 Markdown 表格。"""
        rows = []
        for row in table.rows:
            cells = [(cell.text or "").replace("|", "\\|").replace("\n", " ") for cell in row.cells]
            rows.append(cells)

        if not rows:
            return ""

        md_lines = []
        md_lines.append("| " + " | ".join(rows[0]) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        for row in rows[1:]:
            # 补齐列数
            while len(row) < len(rows[0]):
                row.append("")
            md_lines.append("| " + " | ".join(row[:len(rows[0])]) + " |")

        return "\n".join(md_lines)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    node = DocxToMdNode()
    state = {
        "docx_path": r"D:\test.docx",
        "file_dir": r"D:\output"
    }
    result = node.process(state)
    print(result)
