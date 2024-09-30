# config.py

import os

# Elasticsearch Configuration
ES_USER = "elastic"
ES_PASSWORD = "*MogUVM0Fu5pIm6Ku3QM"
ES_URL = "http://210.205.197.18:9200/"
# ES_INDEX = "langchain_index_chunk_cut7_e5_exaone"
ES_INDEX = "langchain_index_full_history"
ES_HISTORY_INDEX = "langchin_history"
# HuggingFace Token
os.environ['HF_TOKEN'] = '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'
