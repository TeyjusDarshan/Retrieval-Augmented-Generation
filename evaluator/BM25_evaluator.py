'''
    Recall@10: 0.9112582781456954
    Recall@3: 0.8333964049195838
    Recall@2: 0.7943235572374645
    Recall@1: 0.7012298959318827
'''

from datasets import load_dataset
from tqdm import tqdm
import math
from Retriever.bm25_retriever import BM25Retriever
from Metrics.metrics_calculator import MetricsCalculator
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run validation on the retriever.")
    parser.add_argument(
        "--recall_at", 
        type=int, 
        required=True, 
        help="The 'K' value for calculating Recall@K (e.g., 2, 3, 5, 15)"
    )
    args = parser.parse_args()


    ds = load_dataset("rajpurkar/squad")['validation']

    batch_size = 4
    total_batches = math.ceil(len(ds) / batch_size)

    retreiver = BM25Retriever("./indices/bm25_wikipedia_index")
    metricsCalculator = MetricsCalculator() 

    recall_k = 0

    for batch in tqdm(ds.iter(batch_size=batch_size),
                        total = total_batches, 
                        desc = 'Evaluating Retrieval'):
        
        queries = batch["question"]

        results, scores = retreiver.retrieve(queries, args.recall_at)
                
        
        for i in range(len(queries)):
            results_docs = []
            # print("Question: ", queries[i])
            for j in range(len(results[i])):
                # print("Doc ", results[i][j]['text'])
                # print('-' * 40)
                results_docs.append(results[i][j]['text'])
            # print('\n', '+'*40)
                

            recall_k += metricsCalculator.RecallAtK(results_docs, batch["context"][i])

    recall_k_avg = float(recall_k/len(ds))

    print("Recall At K: ", recall_k_avg)


    

    
    
    


