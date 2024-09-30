import os
import re
from tqdm import tqdm
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLineHorizontal
from translation.translate import pdf_translate, translate_model_call
from document_processing.document_utils import save_all_documents_to_single_json, update_documents_with_translation, Document

def find_pdf_files(dir_path):
    file_list = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def extract_main_text(page_layout):
    main_text = ""
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            for text_line in element:
                if isinstance(text_line, LTTextLineHorizontal):
                    font_size = text_line.height
                    if 9 < font_size < 11:
                        text = text_line.get_text()
                        text = re.sub(r'\d+\)', '', text)
                        text = text.replace('\n', ' ').replace('\uf000', '').replace('《', '').replace('》', '')
                        main_text += text
    return main_text.strip()

def extract_all_pages(pdf_file):
    full_text = ""
    for page_layout in extract_pages(pdf_file):
        main_text = extract_main_text(page_layout)
        full_text += main_text + " "
    return full_text.strip()

def split_into_paragraphs_by_period(text):
    sentences = text.split('.')
    paragraphs = []
    paragraph = []

    for i, sentence in enumerate(sentences):
        paragraph.append(sentence.strip())
        if (i + 1) % 7 == 0:
            paragraphs.append('. '.join(paragraph) + '.')
            paragraph = []
    
    if paragraph:
        paragraphs.append('. '.join(paragraph) + '.')
    
    return paragraphs

def process_multiple_pdfs(pdf_files):
    pages = []
    tokenizer, model = translate_model_call()
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        full_text = extract_all_pages(pdf_file)
        paragraphs = split_into_paragraphs_by_period(full_text)

        for i, paragraph in enumerate(tqdm(paragraphs, desc="Translating and saving paragraphs", leave=False)):
            translated_paragraph = pdf_translate(model, tokenizer, paragraph)
            document = Document(metadata={'source': pdf_file, 'paragraph_number': i + 1}, page_content=translated_paragraph)
            pages.append(document)
    
    return pages

def process_pdfs_and_save(dir_path):
    pdf_files = find_pdf_files(dir_path)
    all_documents = process_multiple_pdfs(pdf_files)
    save_all_documents_to_single_json(all_documents, output_file="all_documents.json")
    return all_documents

def process_and_translate_documents(dir_path):
    all_documents = process_pdfs_and_save(dir_path)
    translated_documents = update_documents_with_translation(all_documents)
    return translated_documents
