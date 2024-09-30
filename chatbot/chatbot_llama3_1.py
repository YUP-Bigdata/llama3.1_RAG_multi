import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer,BitsAndBytesConfig
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings
from config import ES_USER, ES_PASSWORD, ES_URL, ES_INDEX, ES_HISTORY_INDEX
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from util.util import Util
from document_processing.document_utils import contains_chinese, hanja_translate
from datetime import datetime
import time
import warnings

# 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, message="The method `BaseRetriever.get_relevant_documents` was deprecated")
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization")

# RerankRetrieverWrapper 클래스 정의
class RerankRetrieverWrapper:
    def __init__(self, vectorstore_retriever, util_instance):
        self.vectorstore_retriever = vectorstore_retriever
        self.util_instance = util_instance

    def __call__(self, inputs):
        query = inputs["question"] if isinstance(inputs, dict) else inputs
        search_results = self.vectorstore_retriever.invoke(query)
        reranked_data = self.util_instance.rerank(query, search_results)
        return {"context": reranked_data, "question": query}

# 히스토리 관리를 위한 클래스
from elasticsearch import Elasticsearch
from datetime import datetime
import time

class ConversationHistory:
    def __init__(self, es_history_index, es_url, es_user, es_password):
        self.es_client = Elasticsearch(
            es_url,
            basic_auth=(es_user, es_password)
        )
        self.es_history_index = es_history_index
        self.create_index_if_not_exists()  # 인덱스가 없으면 생성

    def create_index_if_not_exists(self):
        # 인덱스가 존재하는지 확인
        if not self.es_client.indices.exists(index=self.es_history_index):
            # 인덱스 설정 (필요에 따라 매핑과 설정을 추가할 수 있습니다)
            index_settings = {
                "mappings": {
                    "properties": {
                        "session_id": {"type": "keyword"},
                        "question": {"type": "text"},
                        "answer": {"type": "text"},
                        "timestamp": {"type": "date"}
                    }
                }
            }
            # 인덱스 생성
            self.es_client.indices.create(index=self.es_history_index, body=index_settings)

    def save_history(self, session_id, question, answer):
        timestamp_iso = datetime.fromtimestamp(time.time()).isoformat()
        document = {
            'session_id': session_id,
            'question': question,
            'answer': answer,
            'timestamp': timestamp_iso  
        }
        self.es_client.index(index=self.es_history_index, body=document)

    def retrieve_history(self, session_id, limit=5):
        query = {
            "query": {
                "match": {"session_id": session_id}
            },
            "size": limit,
            "sort": [
                {"timestamp": {"order": "desc"}}
            ]
        }
        results = self.es_client.search(index=self.es_history_index, body=query)
        return [(hit['_source']['question'], hit['_source']['answer']) for hit in results['hits']['hits']]


# 벡터 저장소 호출 함수
def vectorstore_call(model_name, model_kwargs, encode_kwargs):
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return ElasticsearchStore(
        index_name=ES_INDEX,
        embedding=embedding_model,
        es_url=ES_URL,
        es_user=ES_USER,
        es_password=ES_PASSWORD
    )

# 커스텀 파서
# class CustomOutputParser(BaseOutputParser):
#     def parse(self, text: str) -> str:
#         split_text = text.split('<|start_header_id|>assistant<|end_header_id|>\n', 1)
#         if len(split_text) > 1:
#             result = split_text[1].strip()
#             while contains_chinese(result):
#                 result = hanja_translate(result)
#             return result
#         else:
#             return text

class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Split the text to find the part after "Answer:"
        split_text = text.split('Answer:', 1)
        if len(split_text) > 1:
            # Extract the part after "Answer:" and strip any extra whitespace
            result = split_text[1].split(' [/INST]', 1)[0].strip()
            return result
        else:
            # If "Answer:" isn't found, return the original text
            return text


# 챗봇 실행 설정 함수
def setup_and_run_chatbot():
    # ElasticsearchStore 설정
    model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {'device': 0}
    encode_kwargs = {'normalize_embeddings': False}
    
    vectorstore = vectorstore_call(model_name, model_kwargs, encode_kwargs)

    # 챗봇 모델 설정
    model_id = "NCSOFT/Llama-VARCO-8B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,  # Ensuring the 8-bit quantization is used
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.gradient_checkpointing_enable()

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    repetition_penalty=1.2,
    streamer=streamer,
    temperature=0.7,  # Reduced for more fluent output
    top_p=0.9         # Nucleus sampling for better coherence
    )


    hf_pipeline = HuggingFacePipeline(pipeline=pipe)

    # Prompt Template
    temp = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 당신은 친절한 어시스턴트입니다. 문맥과 이전 기록에 있는 정보만을 참고해 질문에 답변하세요. 답변은 한국어로 작성하세요. 답변 외에 다른 정보는 필요하지 않습니다. 만약 문맥에 있는 정보로 답변할 수 없다면 "모르겠습니다."라고 답변하세요. <|eot_id|><|start_header_id|>user<|end_header_id|> Question: {question}
Context: {context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer:
"""

    prompt_template = PromptTemplate.from_template(temp)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

    util_instance = Util()

    rerank_wrapper = RerankRetrieverWrapper(retriever, util_instance)

    rag_chain = {
        "context": rerank_wrapper,
        "question": RunnablePassthrough()
    } | prompt_template | hf_pipeline | CustomOutputParser()

    conversation_history = ConversationHistory(ES_HISTORY_INDEX, ES_URL, ES_USER, ES_PASSWORD)
    
    # 대화가 시작될 때 한 번만 session_id 생성
    session_id = str(time.time())  # 대화가 끝날 때까지 유지되는 session_id

    while True:
        question = input("\n 질문을 입력하세요 ('exit' 입력 시 종료): ")
        if question.lower() == 'exit':
            break
        elif not question.strip():
            print("질문이 입력되지 않았습니다. 다시 입력해주세요.")
            continue
        
        # 히스토리 불러오기 - 가장 최근 3개의 기록만 불러오기
        history_data = conversation_history.retrieve_history(session_id, limit=3)
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history_data])

        # 프롬프트에 히스토리 추가
        prompt_with_history = f"Hisroty: {history_text}\n\nQuestion: {question}\nAnswer: "
        
        start_time = time.time() 
        # 질문과 히스토리를 포함한 context를 랭킹하고, RAG 체인으로 답변 생성
        response = rag_chain.invoke({"question": question, "context": prompt_with_history})

        # 응답 저장
        conversation_history.save_history(session_id, question, response)
        
        print('작업 수행된 시간 : %f 초' % (time.time() - start_time))
        # print("답변:", response)

