import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch import nn
from transformers import get_linear_schedule_with_warmup
from Models.retreiver_model import SymmetricBiEncoder
import os








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
batch_size = 16
learning_rate = 5e-5
num_epochs = 3
train_set = 0.8
val_set = 0.2
best_val_loss = float('inf')
best_val_loss_step = 0
patience = 5
patience_counter = 0
global_step = 0
eval_every_steps = 500
checkpoint_dir = "./best_model_checkpoint"



tokenizer = AutoTokenizer.from_pretrained(model_name)


ds = load_dataset("rajpurkar/squad")


class WikiDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        sample = self.ds[index]
        return {
            'question' : sample['question'],
            'context' : sample['context']
        }

    
def collate_fn(batch):
    questions = [item["question"] for item in batch]
    context = [item['context'] for item in batch]

    tokenized_questions = tokenizer(
        questions, 
        padding='longest',
        return_tensors = 'pt'
    )

    tokenized_context = tokenizer(
        context, 
        padding='longest',
        return_tensors = 'pt'
    )

    tokenized_questions = {k: v.to(device) for k, v in tokenized_questions.items()}
    tokenized_context = {k: v.to(device) for k, v in tokenized_context.items()}

    return tokenized_questions, tokenized_context





criterion = nn.CrossEntropyLoss()

# Extract and shuffle the training split properly
train_split = ds['train'].shuffle(seed=42)
ds_size = len(train_split)

train_end_idx = int(ds_size * train_set)

# Slice into subsets using .select()
train_ds = train_split.select(range(0, train_end_idx))
val_ds = train_split.select(range(train_end_idx, ds_size))

train_dataloader = DataLoader(WikiDataset(train_ds), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(WikiDataset(val_ds), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

bi_encoder = SymmetricBiEncoder(model_name, device)

optimizer = torch.optim.AdamW(
    bi_encoder.parameters(), 
    lr=learning_rate,
    weight_decay=0.01
)

total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.10 * total_steps) # 10% of training will be warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)




for epoch in range(num_epochs):
    
    for q_batch, c_batch in tqdm(train_dataloader, desc="training Batches", unit="batch"):
        bi_encoder.train()
        global_step += 1

        optimizer.zero_grad()

        c_embeddings = bi_encoder(c_batch)
        q_embeddings = bi_encoder(q_batch)

        temperature = 0.05
        similarity_matrix = q_embeddings @ c_embeddings.transpose(-1, -2) # Shape: [B, B]
        logits = similarity_matrix / temperature

        targets = torch.arange(logits.size(0), dtype=torch.long, device=device)

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        scheduler.step()

        if global_step%eval_every_steps == 0:
            bi_encoder.eval()
            validation_loss = 0
            size = 0
            with torch.no_grad():
                for q_val_batch, c_val_batch in tqdm(val_dataloader, desc="validating Batches", unit="batch"):

                    c_embeddings = bi_encoder(c_val_batch)
                    q_embeddings = bi_encoder(q_val_batch)

                    temperature = 0.05
                    similarity_matrix = q_embeddings @ c_embeddings.transpose(-1, -2) # Shape: [B, B]
                    logits = similarity_matrix / temperature

                    targets = torch.arange(logits.size(0), dtype=torch.long, device=device)

                    loss = criterion(logits, targets)

                    targets = torch.arange(logits.size(0), dtype=torch.long, device=device)

                    loss = criterion(logits, targets)

                    validation_loss += loss.item() * logits.size(0)
                    size += logits.size(0)
                
                avg_val_loss = validation_loss/size
                print(f"validation loss for epoch {epoch}: {avg_val_loss}")


                if(avg_val_loss < best_val_loss):
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    bi_encoder.model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': bi_encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, os.path.join(checkpoint_dir, "training_state.pt"))
                else:
                    patience_counter += 1
                    print(f" No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("🛑 Early stopping triggered! Training terminated.")
                        break
    # This break handles exiting the outer epoch loop if inner loop breaks
    if patience_counter >= patience:
        break

print("Training complete!")





        



