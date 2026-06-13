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
    ds = load_dataset("rajpurkar/squad")
    context = list(set(ds['train']['context'][:50]))

    print("unique context" , len(context))
    wikidataset = WikiDataset(context)

    chroma_client = chromadb.PersistentClient(path='./RAG_db')
    collection = chroma_client.get_or_create_collection(name='baseline_wiki_embeddings')

    batch_size = 8
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
            
            embeddings_list = mean_pooled.cpu().tolist()

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