import os, sys, torch,gc
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import kneighbors_graph
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from imblearn.over_sampling import RandomOverSampler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, LoRA_config, load_and_prepare_all_data, compute_metrics, create_out_dir

class GCNClassifier(torch.nn.Module):
    """
    Graph Convolutional Network for node classification.
    It consists of two GCN layers.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        # Output raw logits for CrossEntropyLoss
        return x

def build_graph(embeddings, k):
    """
    Constructs a k-NN graph from node embeddings.
    """
    print(f"Building k-NN graph with k={k}...")
    # Using scikit-learn to build the adjacency matrix efficiently
    adj_matrix = kneighbors_graph(embeddings, k, mode='connectivity', include_self=False)
    # Convert the sparse adjacency matrix to PyTorch Geometric's edge_index format
    edge_index = from_scipy_sparse_matrix(adj_matrix)[0]
    return edge_index

def get_embeddings(model, dataset, tokenizer, device):
    """
    Extracts mean-pooled embeddings from the last hidden state of the Llama model.
    """
    model.eval()
    all_embeddings = []
    
    # Create a temporary trainer to use its dataloader
    temp_trainer = Trainer(model=model, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    dataloader = temp_trainer.get_eval_dataloader(dataset)

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to the correct device
            batch = {k: v.to(device) for k, v in batch.items()}
            # We need the hidden states, not the classification logits
            outputs = model.base_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True
            )
            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            # Mean pooling: average the embeddings across the sequence length
            # We use the attention mask to ignore padding tokens
            mask = batch['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            all_embeddings.append(mean_pooled.cpu())

    return torch.cat(all_embeddings, dim=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"Using Model: {Config.MODEL_CHECKPOINT}")

print("Loading tokenizer...")
tokenizer_llama = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT, token=Config.get_hf_token())
if tokenizer_llama.pad_token is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token

trainval_df, test1_df, test2_df = load_and_prepare_all_data()
# for target_trait in Config.ALL_TARGET_COLUMNS:
for target_trait in ['cEXT', 'cOPN']:
    print("\n" + "="*80)
    print(f" PROCESSING TRAIT: {target_trait} ".center(80, "="))
    print("="*80)

    print(f"\nBalancing training data for class labels in trait: {target_trait}")
    train_df, val_df = train_test_split(trainval_df, test_size=0.1, random_state=42, stratify=trainval_df[target_trait])
    # X_train = train_df.drop(columns=Config.ALL_TARGET_COLUMNS)
    # y_train = train_df[target_trait]
    # ros = RandomOverSampler(random_state=42)
    # X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    # train_df_balanced = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    
    base_dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test1': Dataset.from_pandas(test1_df),
        'test2': Dataset.from_pandas(test2_df)
    })
    
    trait_dataset_dict = base_dataset_dict.rename_column(target_trait, "label")
    def tokenize_function(examples):
        return tokenizer_llama(examples["text"], truncation=True, max_length=Config.TOKEN_MAX_LEN)
    tokenized_dataset = trait_dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # ---------------------------------------------------------------------------------
    # STAGE 1: FINE-TUNE LLAMA-LORA TO GET GOOD EMBEDDINGS
    # ---------------------------------------------------------------------------------
    print("\n--- STAGE 1: Fine-tuning Llama-LoRA ---")
    model_llama = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model_llama.config.pad_token_id = tokenizer_llama.pad_token_id
    model_llama_lora = get_peft_model(model_llama, LoRA_config.lora_config)
    model_llama_lora.print_trainable_parameters()

    training_args = LoRA_config.training_args_template
    training_args.output_dir = create_out_dir(target_trait)

    trainer_llama = Trainer(
        model=model_llama_lora,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_llama),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer_llama.train()

    # ---------------------------------------------------------------------------------
    # STAGE 2: GCN CLASSIFICATION ON LLAMA EMBEDDINGS
    # ---------------------------------------------------------------------------------
    print("\n--- STAGE 2: GCN Training and Evaluation ---")
    
    # --- Step 2.1: Extract Embeddings from fine-tuned Llama ---
    print("Extracting embeddings for all datasets...")
    # The trainer loads the best model, which we will use
    fine_tuned_model = trainer_llama.model
    
    # We create one large dataset to build the graph, then create masks
    full_dataset_df = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='validation'),
        test1_df.assign(split='test1'),
        test2_df.assign(split='test2')
    ]).reset_index(drop=True)
    
    full_dataset_tokenized = Dataset.from_pandas(full_dataset_df).map(
        tokenize_function, batched=True, remove_columns=list(full_dataset_df.columns)
    )

    all_embeddings = get_embeddings(fine_tuned_model, full_dataset_tokenized, tokenizer_llama, device)
    labels = torch.tensor(full_dataset_df[target_trait].values, dtype=torch.long)
    
    # Create masks for train, validation, and test sets
    train_mask = torch.tensor(full_dataset_df['split'] == 'train')
    val_mask = torch.tensor(full_dataset_df['split'] == 'validation')
    test1_mask = torch.tensor(full_dataset_df['split'] == 'test1')
    test2_mask = torch.tensor(full_dataset_df['split'] == 'test2')

    # --- Step 2.2: Build the Graph ---
    # The graph structure is built using ALL embeddings to capture global relationships
    edge_index = build_graph(all_embeddings.numpy(), k=Config.GRAPH_NEIGHBORS)
    graph_data = Data(x=all_embeddings, edge_index=edge_index, y=labels)
    graph_data = graph_data.to(device)

    # --- Step 2.3: Train the GCN ---
    print("Training GCN classifier...")
    gcn_model = GCNClassifier(
        in_channels=graph_data.num_features,
        hidden_channels=Config.GCN_HIDDEN_CHANNELS,
        out_channels=2 # Binary classification
    ).to(device)

    optimizer = AdamW(gcn_model.parameters(), lr=Config.GCN_LR, weight_decay=Config.GCN_WEIGHT_DECAY)
    criterion = CrossEntropyLoss()

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(Config.GCN_EPOCHS):
        gcn_model.train()
        optimizer.zero_grad()
        out = gcn_model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        gcn_model.eval()
        with torch.no_grad():
            out = gcn_model(graph_data.x, graph_data.edge_index)
            preds = out.argmax(dim=1)
            
            val_f1 = f1_score(graph_data.y[val_mask].cpu(), preds[val_mask].cpu())
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save the best model state
                torch.save(gcn_model.state_dict(), os.path.join(Config.BASE_OUTPUT_DIR, target_trait, 'best_gcn_model.pt'))
            else:
                patience_counter += 1

        if patience_counter >= Config.GCN_EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # --- Step 2.4: Evaluate the GCN ---
    print("Evaluating GCN on test sets...")
    # Load the best model for final evaluation
    gcn_model.load_state_dict(torch.load(os.path.join(Config.BASE_OUTPUT_DIR, target_trait, 'best_gcn_model.pt')))
    gcn_model.eval()

    with torch.no_grad():
        final_out = gcn_model(graph_data.x, graph_data.edge_index)
        final_preds = final_out.argmax(dim=1)
        
        for split_name, mask in [('test1', test1_mask), ('test2', test2_mask)]:
            test_labels = graph_data.y[mask].cpu().numpy()
            test_preds = final_preds[mask].cpu().numpy()
            
            acc = accuracy_score(test_labels, test_preds)
            f1 = f1_score(test_labels, test_preds, average='binary')
            
            print("\n" + "-"*50)
            print(f"GCN Test Results on {split_name} for trait: {target_trait}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print("-"*50)

    # --- Cleanup ---
    del model_llama, model_llama_lora, trainer_llama, gcn_model, graph_data
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*80)
print(" SCRIPT FINISHED: ALL TRAITS PROCESSED ".center(80, "="))
print("="*80)
