import fitz
from collections import defaultdict

class DocumentParser:
    """A context-aware PDF parser that structures content by headings.

    This parser uses font size heuristics to identify headings within a PDF
    document. It then groups the subsequent text under these headings,
    creating larger, more semantically meaningful chunks of content than
    simple fixed-size splitting would allow.
    """

    def parse(self, file_path: str) -> list[dict]:
        """Extracts text from a PDF, organizing it into sections.

        Args:
            file_path: The local path to the PDF file to be parsed.

        Returns:
            A list of dictionaries, where each dictionary represents a
            semantically grouped section of the document, containing the
            page number, the identified heading, and the associated content.
            Returns an empty list if the document contains no text.
        """
        doc = fitz.open(file_path)
        parsed_sections = []
        current_heading = ""
        current_text = ""
        page_num = 1
        
        font_counts = defaultdict(int)
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_counts[round(span["size"])] += 1
        
        if not font_counts:
            return []
            
        most_common_size = max(font_counts, key=font_counts.get)

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        spans = line["spans"]
                        if not spans: continue

                        span_size = round(spans[0]["size"])
                        line_text = "".join([span["text"] for span in spans]).strip()

                        if span_size > most_common_size:
                            if current_heading or current_text:
                                parsed_sections.append({
                                    "page_number": page_num,
                                    "heading": current_heading,
                                    "content": current_text.strip()
                                })
                            current_heading = line_text
                            current_text = ""
                        else:
                            current_text += " " + line_text
        
        if current_heading or current_text:
            parsed_sections.append({
                "page_number": page_num,
                "heading": current_heading,
                "content": current_text.strip()
            })

        return [sec for sec in parsed_sections if sec["content"]]