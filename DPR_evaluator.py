'''
    Findings:
        Without fine tuning the Modernbert embeddings gives us a Average Recall@K of 0. 
'''


from transformers import AutoModel, AutoTokenizer
import torch
from Retriever.dense_passage_retriever import DensePassageRetreiver
from datasets import load_dataset
from Metrics.metrics_calculator import MetricsCalculator
import numpy as np
from tqdm import tqdm
import math


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

    
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    ds = load_dataset("rajpurkar/squad")['validation']
    retreiver = DensePassageRetreiver(model, tokenizer, './RAG_db', 'baseline_wiki_embeddings')
    metricsCalculator = MetricsCalculator()
    batch_size = 4
    total_batches = math.ceil(len(ds) / batch_size)


    print("Starting to Run Validation on the retreiver")
    recall_k = 0.0
    for batch in tqdm(ds.iter(batch_size=batch_size),
                      total = total_batches, 
                      desc = 'Evaluating Retrieval'):
        queries = batch["question"]

    # queries = ["To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"]
        docs = retreiver.retrieve(queries, 15)['documents']
        
        for i in range(len(queries)):
            recall_k += metricsCalculator.RecallAtK(docs[i], batch["context"][i])
    
    recall_k_avg = float(recall_k/len(ds))

    print("Recall At K: ", recall_k_avg)
