import sys
import os 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from transformers import AutoModel
import torch 
from torch import nn

class SymmetricBiEncoder(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def forward(self, c_input, q_input):
        q_outputs = self.model(**q_input)
        c_outputs = self.model(**c_input)

        q_embeddings = self.mean_pooling(q_outputs, q_input['attention_mask'])
        c_embeddings = self.mean_pooling(c_outputs, c_input['attention_mask'])

        q_embeddings = torch.nn.functional.normalize(q_embeddings, p=2, dim=-1)
        c_embeddings = torch.nn.functional.normalize(c_embeddings, p=2, dim=-1)

        similarity_matrix = q_embeddings @ c_embeddings.transpose(-1, -2)

        temperature = 0.05 
        scaled_similarity = similarity_matrix / temperature

        
        return scaled_similarity # shape [B, num questions, num context]
    

    def mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the last hidden state of the model while ignoring padding tokens.
        """
        token_embeddings = model_output.last_hidden_state 

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask