'''
Findings: 

    Maximum token size of context is 892

'''

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  
import argparse
import torch.nn.functional as F





if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍏 Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")




class WikiDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        inputs = tokenizer(sample, padding="max_length", truncation=True, max_length=892, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in inputs.items()}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate embeddings using a fine-tuned model and store them in ChromaDB.")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="./model_weights/best_model_checkpoint", 
        help="Path to the saved model checkpoint or Hugging Face model ID."
    )
    parser.add_argument(
        "--db_path", 
        type=str, 
        default="./indices/RAG_db_2", 
        help="Directory path where ChromaDB persistent data will be stored."
    )
    parser.add_argument(
        "--collection_name", 
        type=str, 
        default="finetuned_wiki_embeddings", 
        help="Name of the collection inside ChromaDB."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size used for running model inference."
    )

    args = parser.parse_args()

    model_name = args.model_name

    print(f"Loading tokenizer and model from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    ds = load_dataset("rajpurkar/squad")
    context = list(set(ds['validation']['context']))

    print("unique context" , len(context))
    wikidataset = WikiDataset(context)

    print(f"Initializing ChromaDB at '{args.db_path}' with collection '{args.collection_name}'")
    chroma_client = chromadb.PersistentClient(path=args.db_path)
    collection = chroma_client.get_or_create_collection(name=args.collection_name)

    batch_size = args.batch_size
    data_loader = DataLoader(
        dataset=wikidataset, 
        batch_size=batch_size,
        shuffle=False
    )

    print("data loader size ", len(data_loader))

    print("\n🚀 Starting embedding generation and Chroma storage...")

    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing Batches", unit="batch")):

            batch = {key: val.to(device) for key, val in batch.items()}
            
            outputs = model(**batch)
            

            last_hidden = outputs.last_hidden_state
        
            attention_mask = batch['attention_mask'].unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
            
            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            normalized_docs = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            embeddings_list = normalized_docs.cpu().tolist()

            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_contexts = context[start_idx:end_idx]
            batch_ids = [str(i) for i in range(start_idx, min(end_idx, len(context)))]
            batch_metadatas = [{"source": "squad_dataset"} for _ in batch_contexts]

            collection.add(
                embeddings=embeddings_list,
                documents=batch_contexts,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
            
    print("\n✅ All documents have been successfully vectorized and stored in Chroma DB!")