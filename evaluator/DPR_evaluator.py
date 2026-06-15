'''
    Recall@15: 0.9299905392620624
    Recall@10: 0.9011352885525071
    Recall@3: 0.7628192999053927

'''


from transformers import AutoModel, AutoTokenizer
import torch
from Retriever.dense_passage_retriever import DensePassageRetreiver
from datasets import load_dataset
from Metrics.metrics_calculator import MetricsCalculator
import numpy as np
from tqdm import tqdm
import math
import argparse  # Added for CLI argument parsing


if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Run validation on the retriever.")
    parser.add_argument(
        "--recall_at", 
        type=int, 
        required=True, 
        help="The 'K' value for calculating Recall@K (e.g., 2, 3, 5, 15)"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    
    model_name = "Teyjus/modernbert-squad-finetuned"
    # model_name = 'answerdotai/ModernBERT-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    ds = load_dataset("rajpurkar/squad")['validation']
    retreiver = DensePassageRetreiver(model, tokenizer, './indices/RAG_db_2', 'finetuned_wiki_embeddings')
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
        # Passed args.recall_at dynamically to the retriever instead of the hardcoded 3
        docs = retreiver.retrieve(queries, args.recall_at)['documents']
        
        for i in range(len(queries)):
            recall_k += metricsCalculator.RecallAtK(docs[i], batch["context"][i])
    
    recall_k_avg = float(recall_k/len(ds))

    print(f"Recall At {args.recall_at}: ", recall_k_avg)