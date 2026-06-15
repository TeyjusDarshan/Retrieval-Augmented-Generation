'''
    Recall@15: 0.9718070009460738
    Recall@10: 0.9604541154210028
    Recall@5: 0.9285714285714286
    Recall@3: 0.8990539262062441
    Recall@1: 0.8076631977294229

'''

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModel, AutoTokenizer
import torch
from datasets import load_dataset
from Metrics.metrics_calculator import MetricsCalculator
from tqdm import tqdm
import math
from Retriever.hybrid_retreiver import HybridRetreiver
import argparse


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    parser = argparse.ArgumentParser(description="Run validation on the retriever.")
    parser.add_argument(
        "--recall_at", 
        type=int, 
        required=True, 
        help="The 'K' value for calculating Recall@K (e.g., 2, 3, 5, 15)"
    )
    args = parser.parse_args()

    
    model_name = "Teyjus/modernbert-squad-finetuned"
    # model_name = 'answerdotai/ModernBERT-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    ds = load_dataset("rajpurkar/squad")['validation']
    metricsCalculator = MetricsCalculator()

    hybridRetreiver = HybridRetreiver(model, tokenizer, './indices/RAG_db_2', 'finetuned_wiki_embeddings', "./indices/bm25_wikipedia_index", 60)


    batch_size = 4
    total_batches = math.ceil(len(ds) / batch_size)
    recall_k = 0

    for batch in tqdm(ds.iter(batch_size=batch_size),
                      total = total_batches, 
                      desc = 'Evaluating Retrieval'):
        queries = batch["question"]
        results = hybridRetreiver.retrieve(queries, args.recall_at)
        for i in range(len(queries)):
            recall_k += metricsCalculator.RecallAtK(results[i], batch["context"][i])

    recall_k_avg = float(recall_k/len(ds))

    print("Recall At K: ", recall_k_avg)




