import os, gc, sys
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, load_and_prepare_all_data, compute_mtl_metrics


class LoRA_Config:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    training_args = TrainingArguments(
        output_dir=Config.BASE_OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        weight_decay=0.05,
        lr_scheduler_type="linear",
        warmup_steps=200,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_total_limit=1,
        disable_tqdm=False,
        label_names=["labels"],
    )

class GCN_Config:
    """Configuration class for the GCN Stage."""
    GRAPH_NEIGHBORS = 10
    GCN_HIDDEN_CHANNELS = 256
    GCN_EPOCHS = 200
    GCN_LR = 1e-3
    GCN_WEIGHT_DECAY = 5e-4
    GCN_EARLY_STOPPING_PATIENCE = 10


class GCNClassifier(torch.nn.Module):
    """A simple two-layer GCN for multi-label node classification."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def build_graph(embeddings, k):
    """Constructs a k-NN graph from node embeddings."""
    print(f"Building k-NN graph with k={k}...")
    adj_matrix = kneighbors_graph(embeddings, k, mode='connectivity', include_self=False)
    edge_index = from_scipy_sparse_matrix(adj_matrix)[0]
    return edge_index

def get_embeddings(model, dataset, tokenizer, device):
    """Extracts mean-pooled embeddings from the Llama model."""
    print("Extracting embeddings from the fine-tuned Llama model...")
    model.eval()
    all_embeddings = []
    
    temp_trainer = Trainer(model=model, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    dataloader = temp_trainer.get_eval_dataloader(dataset)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            outputs = model.base_model.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            mask = batch['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            all_embeddings.append(mean_pooled.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using Model: {Config.MODEL_CHECKPOINT}")
    print('Using args', LoRA_Config.training_args)


    tokenizer_llama = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT, token=Config.get_hf_token())
    if tokenizer_llama.pad_token is None:
        tokenizer_llama.pad_token = tokenizer_llama.eos_token

    trainval_df, test1_df, test2_df = load_and_prepare_all_data(0.15)
    train_df, val_df = train_test_split(trainval_df, test_size=0.15, random_state=42)
    print(f"\nData split into {len(train_df)} train, {len(val_df)} validation, {len(test1_df)} test1, and {len(test2_df)} test2 samples.")
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test1': Dataset.from_pandas(test1_df),
        'test2': Dataset.from_pandas(test2_df)
    })
    
    def tokenize_function(examples):
        tokenized_inputs = tokenizer_llama(examples['text'], truncation=True, max_length=128)
        labels = np.array([examples[col] for col in Config.ALL_TARGET_COLUMNS])
        tokenized_inputs['labels'] = labels.T.astype(np.float32).tolist()
        return tokenized_inputs
    
    tokenized_dataset = dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS)

    print("\n--- STAGE 1: Fine-tuning Llama-LoRA for Multi-Task Learning ---")
    model_llama = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=len(Config.ALL_TARGET_COLUMNS),
        problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model_llama.config.pad_token_id = tokenizer_llama.pad_token_id
    model_llama_lora = get_peft_model(model_llama, LoRA_Config.lora_config)
    model_llama_lora.print_trainable_parameters()

    trainer_llama = Trainer(
        model=model_llama_lora,
        args=LoRA_Config.training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_llama),
        compute_metrics=compute_mtl_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer_llama.train()

    # --- STAGE 2: GCN Classification on MTL Embeddings ---
    print("\n--- STAGE 2: GCN Training and Evaluation for MTL ---")
    fine_tuned_model = trainer_llama.model
    
    full_graph_df = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='validation'),
        test1_df.assign(split='test1')
    ]).reset_index(drop=True)
    
    full_graph_dataset = Dataset.from_pandas(full_graph_df)
    full_graph_tokenized = full_graph_dataset.map(tokenize_function, batched=True, remove_columns=list(full_graph_df.columns))

    all_embeddings = get_embeddings(fine_tuned_model, full_graph_tokenized, tokenizer_llama, device)
    labels = torch.tensor(full_graph_df['labels'].tolist(), dtype=torch.float)
    
    train_mask = torch.tensor(full_graph_df['split'] == 'train')
    val_mask = torch.tensor(full_graph_df['split'] == 'validation')
    test1_mask = torch.tensor(full_graph_df['split'] == 'test1')

    edge_index = build_graph(all_embeddings.numpy(), k=GCN_Config.GRAPH_NEIGHBORS)
    graph_data = Data(x=all_embeddings, edge_index=edge_index, y=labels).to(device)

    print("Training MTL GCN classifier...")
    gcn_model = GCNClassifier(
        in_channels=graph_data.num_features,
        hidden_channels=GCN_Config.GCN_HIDDEN_CHANNELS,
        out_channels=len(Config.ALL_TARGET_COLUMNS)
    ).to(device)

    optimizer = AdamW(gcn_model.parameters(), lr=GCN_Config.GCN_LR, weight_decay=GCN_Config.GCN_WEIGHT_DECAY)
    criterion = BCEWithLogitsLoss()

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(GCN_Config.GCN_EPOCHS):
        gcn_model.train()
        optimizer.zero_grad()
        out = gcn_model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()

        gcn_model.eval()
        with torch.no_grad():
            out = gcn_model(graph_data.x, graph_data.edge_index)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).int()
            val_f1 = f1_score(graph_data.y[val_mask].cpu(), preds[val_mask].cpu(), average='micro')
            
            if (epoch + 1) % 10 == 0:
                print(f'GCN Epoch {epoch+1:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(gcn_model.state_dict(), os.path.join(Config.BASE_OUTPUT_DIR, 'best_gcn_mtl_model.pt'))
            else:
                patience_counter += 1

        if patience_counter >= GCN_Config.GCN_EARLY_STOPPING_PATIENCE:
            print(f"GCN early stopping at epoch {epoch+1}")
            break
    
    print("\nEvaluating MTL GCN on test set...")
    gcn_model.load_state_dict(torch.load(os.path.join(Config.BASE_OUTPUT_DIR, 'best_gcn_mtl_model.pt')))
    gcn_model.eval()

    with torch.no_grad():
        final_out = gcn_model(graph_data.x, graph_data.edge_index)
        final_probs = torch.sigmoid(final_out)
        final_preds = (final_probs > 0.5).int()
        
        test_labels = graph_data.y[test1_mask].cpu().numpy()
        test_preds = final_preds[test1_mask].cpu().numpy()
        
        f1_micro = f1_score(test_labels, test_preds, average='micro')
        roc_auc_micro = roc_auc_score(test_labels, final_probs[test1_mask].cpu().numpy(), average='micro')
        subset_accuracy = accuracy_score(test_labels, test_preds)

        print("\n" + "-"*50)
        print("Final GCN MTL Test Results (Overall):")
        print(f"  Subset Accuracy (Exact Match): {subset_accuracy:.4f}")
        print(f"  F1 Score (micro): {f1_micro:.4f}")
        print(f"  ROC AUC (micro): {roc_auc_micro:.4f}")
        
        print("\nPer-Trait Test Results:")
        for i, trait in enumerate(Config.ALL_TARGET_COLUMNS):
            trait_acc = accuracy_score(test_labels[:, i], test_preds[:, i])
            trait_f1 = f1_score(test_labels[:, i], test_preds[:, i], average='binary')
            print(f"  --- {trait} ---")
            print(f"    Accuracy: {trait_acc:.4f}")
            print(f"    F1 Score: {trait_f1:.4f}")
        print("-"*50)

    del model_llama, model_llama_lora, trainer_llama, gcn_model, graph_data
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
