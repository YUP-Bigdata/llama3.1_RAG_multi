import json
import re
import hanja
from tqdm import tqdm

class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

    def __repr__(self):
        return f"Document(metadata={self.metadata}, page_content={self.page_content})"
    
    def to_dict(self):
        return {
            "metadata": self.metadata,
            "page_content": self.page_content
        }

def save_all_documents_to_single_json(documents, output_file="documents.json"):
    all_docs_dicts = [{
        "metadata": doc.metadata,
        "page_content": doc.page_content
    } for doc in documents]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_docs_dicts, f, ensure_ascii=False, indent=4)
    
    print(f"All documents have been saved to {output_file}")

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def hanja_translate(text):
    return hanja.translate(text, 'substitution')

def update_documents_with_translation(documents):
    for doc in tqdm(documents, desc="Translating documents"):
        if contains_chinese(doc.page_content):
            translated_text = hanja_translate(doc.page_content)
            if translated_text:
                doc.page_content = translated_text
    return documents
