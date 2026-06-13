import sys
import os 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from transformers import AutoModel
import torch 
from torch import nn

class DocumentEncoder(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def forward(self, input):
        outputs = self.model(**input)

        embeddings = self.mean_pooling(outputs, input['attention_mask'])

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings
    

    def mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the last hidden state of the model while ignoring padding tokens.
        """
        token_embeddings = model_output.last_hidden_state 

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask