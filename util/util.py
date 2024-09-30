from FlagEmbedding import FlagReranker

class Util:
    def __init__(self):
        self.reranker = None

    def rerank(self, question, search_results):
        if self.reranker is None:
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        search_data = []
        for doc in search_results:
            data = {
                'Source': doc.metadata['source'],
                'Paragraph Number': doc.metadata['paragraph_number'],
                'Content': doc.page_content
            }
            search_data.append([question, str(data)])
        
        # 점수 계산
        scores = self.reranker.compute_score(search_data)
        
        # 음수 점수를 제외하고 정렬
        # filtered_results = [(score, search_results[i]) for i, score in enumerate(scores) if score > 0]
        filtered_results = [(score, search_results[i]) for i, score in enumerate(scores)]
        filtered_results.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 5개 추출 (부족할 경우 해당 수만큼 반환)
        top_results = [result[1] for result in filtered_results[:5]]
        
        if not top_results:
            # 관련 문서가 없을 경우 경고 메시지를 추가
            top_results.append({
                'Source': 'System',
                'Paragraph Number': -1,
                'Content': "주의: 관련성이 높은 문서가 충분하지 않아 답변을 하기 어렵습니다."
            })
        
        return top_results
