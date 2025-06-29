import os, gc, sys
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig
from sklearn.neighbors import kneighbors_graph
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, load_and_prepare_all_data, compute_pos_weight, create_unique_dir
from utils.trainer import Trainer


class ModelConfig:
    # LoRA Config
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.1

    GRAPH_NEIGHBORS = 3
    GCN_HIDDEN_CHANNELS = 256
    DROP_OUT = 0.5
    POOLING_METHOD = "max"

    # Training Config
    EPOCHS = 5
    BATCH_SIZE = 4 # This is the per-device batch size
    GRAD_ACCUM_STEPS = 8 # Effective batch size = 32
    LR = 1e-5 # A conservative learning rate for end-to-end training
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 2
    TOKEN_MAX_LEN = 128 # Config.TOKEN_MAX_LEN 512
    OUTPUT_DIR = create_unique_dir('LLAMA_LoRA_MTL_GCN')


class LlamaGCNModel(torch.nn.Module):
    def __init__(self, model_checkpoint, num_labels, hidden_channels=256, drop_out=0.5, pooling_method="max"):
        super().__init__()
        self.pooling_method = pooling_method
        self.llama_base = AutoModel.from_pretrained(model_checkpoint, token=Config.get_hf_token(), torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(
            r=ModelConfig.LORA_R,
            lora_alpha=ModelConfig.LORA_ALPHA,
            lora_dropout=ModelConfig.LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.llama_peft = get_peft_model(self.llama_base, lora_config)
        self.llama_peft.print_trainable_parameters()
        
        # GCN Layers
        emb_dim = self.llama_peft.config.hidden_size
        self.conv1 = GCNConv(emb_dim, emb_dim)
        self.conv2 = GCNConv(emb_dim, emb_dim//2)
        self.conv3 = GCNConv(emb_dim//2, num_labels)
        self.drop_out = drop_out

    def forward(self, input_ids, attention_mask):
        outputs = self.llama_peft(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if self.pooling_method == 'mean':
            last_hidden_state = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            x = sum_embeddings / sum_mask
        elif self.pooling_method == 'max':
            last_hidden_state = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            masked_embeddings = last_hidden_state * mask + (1 - mask) * -1e9
            x = torch.max(masked_embeddings, dim=1).values
        else:
            x = outputs.last_hidden_state[:, -1, :]
        
        # Build graph for current batch
        edge_index = self.build_graph(x.detach().cpu().numpy(), k=ModelConfig.GRAPH_NEIGHBORS)
        edge_index = edge_index.to(x.device)
        
        # GCN Forward
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv3(x, edge_index)
        return x

    @staticmethod
    def build_graph(embeddings, k=5):
        """Build k-NN graph from embeddings"""
        actual_k = min(k, len(embeddings) - 1)
        if actual_k < 1:
            return torch.empty((2, 0), dtype=torch.long)  # Empty graph
        adj = kneighbors_graph(embeddings, actual_k, mode='connectivity', include_self=False)
        adj = adj.maximum(adj.T)  # Make symmetric
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        return edge_index

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using config: {ModelConfig.LR}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT, token=Config.get_hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df, val_df, test1_df, test2_df = load_and_prepare_all_data(0.1)
    print(f"\nData split into {len(train_df)} train, {len(val_df)} validation, and test sets.")

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, max_length=ModelConfig.TOKEN_MAX_LEN)
        labels = np.array([examples[col] for col in Config.ALL_TARGET_COLUMNS])
        tokenized_inputs['labels'] = labels.T.astype(np.float32) #.tolist()
        return tokenized_inputs
    
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df).map(preprocess_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
        'validation': Dataset.from_pandas(val_df).map(preprocess_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
        'test1': Dataset.from_pandas(test1_df).map(preprocess_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
        'test2': Dataset.from_pandas(test2_df).map(preprocess_function, batched=True,remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
    })

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(dataset["train"], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset["validation"], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator)
    test1_loader = DataLoader(dataset['test1'], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator)
    test2_loader = DataLoader(dataset['test2'], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator)

    model = LlamaGCNModel(
        model_checkpoint=Config.MODEL_CHECKPOINT,
        num_labels=len(Config.ALL_TARGET_COLUMNS),
        hidden_channels=ModelConfig.GCN_HIDDEN_CHANNELS,
        drop_out=ModelConfig.DROP_OUT,
        pooling_method=ModelConfig.POOLING_METHOD
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=ModelConfig.LR, weight_decay=ModelConfig.WEIGHT_DECAY)
    pos_weight_tensor = compute_pos_weight(train_df, device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(ModelConfig.EPOCHS):
        train_loss = Trainer.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = Trainer.val_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{ModelConfig.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: { val_res['loss']:.4f}, Val F1: {val_res['f1_macro']:.4f}, Val Acc: {val_res['accuracy']:.4f}")
        if val_res['f1_macro'] > best_val_f1:
            best_val_f1 = val_res['f1_macro']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ModelConfig.OUTPUT_DIR, 'best_e2e_model.pt'))
            print(f"New best model saved with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
        if patience_counter >= ModelConfig.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    
    print("\nEvaluating MTL GCN on test set...")
    model.load_state_dict(torch.load(os.path.join(ModelConfig.OUTPUT_DIR, 'best_e2e_model.pt')))

    Trainer.evaluate(model, test1_loader, device, Config.ALL_TARGET_COLUMNS, 'mypersonality')
    Trainer.evaluate(model, test2_loader, device, Config.ALL_TARGET_COLUMNS, 'essay')
  
    print("\n SCRIPT FINISHED ".center(80, "="))


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
