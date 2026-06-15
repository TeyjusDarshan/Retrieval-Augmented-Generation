'''
    Recall@10: 0.9540208136234626
    Recall@3: 0.8946073793755913
    Recall@2: 0.8583727530747398 
    Recall@1: 0.7999053926206244

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


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    
    model_name = "./model_weights/best_model_checkpoint"
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
        results = hybridRetreiver.retrieve(queries, 10)
        for i in range(len(queries)):
            recall_k += metricsCalculator.RecallAtK(results[i], batch["context"][i])

    recall_k_avg = float(recall_k/len(ds))

    print("Recall At K: ", recall_k_avg)




