import sys 
import os 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from retriever import Retriever
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
import torch.nn.functional as F


class DensePassageRetreiver(Retriever):
    def __init__(self, model, tokenizer, db_path, collection_name):
        self.model = model
        self.tokenizer = tokenizer

        chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🍏 Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU")

    def retrieve(self, queries, k):
        inputs = self.tokenizer(queries, padding="longest", truncation=True, max_length=892, return_tensors='pt').to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state

            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
            
            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            normalized_embeddings = F.normalize(mean_pooled, p=2, dim=1)
            
            embeddings_list = normalized_embeddings.cpu().tolist()

            results = self.collection.query(
                query_embeddings=embeddings_list,
                n_results=k,
                include=["embeddings", "documents", "metadatas", "distances"]
            )

            return results


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

    
    model_name = "Teyjus/modernbert-squad-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)


    retreiver = DensePassageRetreiver(model, tokenizer, './RAG_db', 'baseline_wiki_embeddings')
    queries = ["To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"]
    docs = retreiver.retrieve(queries, 20)['documents']


    for i in range(len(queries)):
        print("Query: ", queries[i])
        print('+' * 100)
        for doc in docs[i]:
            print(doc)
            print('-' * 100)


