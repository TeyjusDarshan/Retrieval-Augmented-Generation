import sys
import os
import heapq

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retriever import Retriever
from bm25_retriever import BM25Retriever
from dense_passage_retriever import DensePassageRetreiver
import numpy as np

class HybridRetreiver(Retriever):
    def __init__(self, model, tokenizer, db_path, collection_name, bm25_index_path, rrf_k):
        self.bm25_retriever = BM25Retriever(bm25_index_path)
        self.dpr = DensePassageRetreiver(model, tokenizer, db_path, collection_name)
        self.rrf_k = rrf_k


    def retrieve(self, queries, k):
        bm25_results, scores = self.bm25_retriever.retrieve(queries, k)

        sorted_indices = np.argsort(-scores, axis=-1)

        sorted_results = np.take_along_axis(bm25_results, sorted_indices, axis=-1)

        bm25_results = []
        for result in sorted_results:
            docs = []
            for doc_dict in result:
                docs.append(doc_dict['text'])
            bm25_results.append(docs)
    

        dpr_results = self.dpr.retrieve(queries, k)['documents']

       
        results = self.calculate_rrf_scores(bm25_results, dpr_results, len(queries))



        results = self.get_top_k(results, k)
        return results


    def get_top_k(self, results, k):
        final_results = []
        for result in results:
            top_k_docs = []
            for key, value in result.items():
                if len(top_k_docs) <= k:
                    heapq.heappush(top_k_docs, (value, key))
                else:
                    if top_k_docs and top_k_docs[0][0] <  value:
                        heapq.heappop(top_k_docs)
                        heapq.heappush(top_k_docs, (value, key))
            inter_res = []
            for doc in top_k_docs:
                inter_res.append(doc[1])
            
            final_results.append(inter_res)
        return final_results
            

    def calculate_rrf_score_internal(self, docs):

        doc_dict = {}
        for i in range(len(docs)):
            rank = i + 1
            score = 1.0/(self.rrf_k + rank)
            doc_dict[docs[i]] = score
        return doc_dict
    
    def calculate_rrf_scores(self,bm25_results, dpr_results, num_queries):
        results = []
        for query_idx in range(num_queries):
            dpr_docs = dpr_results[query_idx]
            bm25_docs = bm25_results[query_idx]
            rrf1 = self.calculate_rrf_score_internal(dpr_docs)
            rrf2 = self.calculate_rrf_score_internal(bm25_docs)

            final_dict = {}
            for key, value in rrf1.items():
                if key not in rrf2:
                    final_dict[key] = value
                else:
                    final_dict[key] = value + rrf2[key]
            
            for key, value in rrf2.items():
                if key not in rrf1:
                    final_dict[key] = value 

            results.append(final_dict)
        return results


            