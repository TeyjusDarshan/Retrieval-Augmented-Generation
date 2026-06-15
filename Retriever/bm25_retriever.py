import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bm25s
import Stemmer
from retriever import Retriever




class BM25Retriever(Retriever):
    def __init__(self, index_path):
        self.retriever = bm25s.BM25.load(index_path, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
    
    def retrieve(self, queries, k):
        query_tokens = bm25s.tokenize(queries, stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        return results, scores


if __name__ == "__main__":
    retreiver = BM25Retriever("bm25_wikipedia_index")
    queries = [
        "What is in front of the Notre Dame Main Building?",
        "What was the amount of wins Knute Rockne attained at Notre Dame while head coach?"
    ]

    k = 5

    results, scores = retreiver.retrieve(queries, k)
    for query_idx, query_text in enumerate(queries):
        print(f"\nQuery: '{query_text}'")
        print("-" * 40)
        
        for rank in range(k):
            # Fetch the document dictionary directly from the results matrix
            doc = results[query_idx][rank]
            score = scores[query_idx][rank]
            
            print(f"Rank {rank + 1} (BM25 Score: {score:.2f})")
            print(f"Doc ID: {doc['id']}")
            print(f"Snippet: {doc['text'][:120]}...")
            print("=" * 20)