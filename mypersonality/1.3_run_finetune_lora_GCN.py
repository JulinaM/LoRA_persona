import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType

# NEW IMPORTS for Graph Convolutional Network
# You may need to install these: pip install torch_geometric torch_sparse torch_scatter
from torch_geometric.nn import GCNConv
from torch.nn import functional as F

## ---------------------------------------------------
## --- Configuration ---
## ---------------------------------------------------
ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
TEXT_COLUMN = "STATUS"
DATA_FILE = "/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv"
MODEL_CHECKPOINT_LLAMA = "meta-llama/Meta-Llama-3-8B"
BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt_llama_gcn_lora" 
my_token = "" # Your Hugging Face token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

## ---------------------------------------------------
## --- Custom GCN Model Definition (Inheritance) ---
## ---------------------------------------------------
class LlamaForGraphClassification(LlamaPreTrainedModel):
    """
    Custom model integrating a Llama backbone with a GCN classifier head.
    This model expects not just text inputs but also graph structure data.
    """
    def __init__(self, config, gcn_hidden_channels=256, dropout_rate=0.3):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config) # The Llama backbone

        # GCN Classifier Head
        self.gcn_conv1 = GCNConv(config.hidden_size, gcn_hidden_channels)
        self.gcn_conv2 = GCNConv(gcn_hidden_channels, self.num_labels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_features, edge_index):
        """
        The forward pass for the GCN head. It takes pre-computed node features.
        """
        x = self.gcn_conv1(node_features, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gcn_conv2(x, edge_index)
        return x

## ---------------------------------------------------
## --- Custom Trainer for Graph Model ---
## ---------------------------------------------------
class GCNTrainer(Trainer):
    """
    Custom Trainer to handle the two-stage forward pass:
    1. Get all node embeddings from the Llama model.
    2. Pass the embeddings and graph structure through the GCN head.
    """
    # THE FIX: Added `num_items_in_batch=None` to match the parent Trainer's method signature.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # The Trainer has already moved the 'inputs' dict to the correct device.
        # Extract graph structure and labels.
        edge_index = inputs.pop("edge_index")
        graph_labels = inputs.pop("graph_labels")
        
        # 1. Get node embeddings from the Llama backbone (the 'model.model' part)
        node_features_outputs = model.model.model(**inputs)
        node_features = node_features_outputs.last_hidden_state
        
        # Perform masked mean pooling on the node features
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(node_features.size()).float()
        sum_embeddings = torch.sum(node_features * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_features = sum_embeddings / sum_mask

        # 2. Pass embeddings and graph structure through the GCN head
        logits = model(node_features=pooled_features, edge_index=edge_index)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, graph_labels)
        
        return (loss, {"logits": logits}) if return_outputs else loss

## ---------------------------------------------------
## --- Data Loading Function (no changes) ---
## ---------------------------------------------------
def load_and_prepare_data(file_path, text_col, target_col):
    print(f"\n--- Loading and preparing data for target: {target_col} ---")
    df = pd.read_csv(file_path, encoding='Windows-1252')
    df = df.dropna(subset=[text_col, target_col])
    
    if df[target_col].nunique() < 2:
        print(f"Warning: Not enough class diversity for {target_col}. Skipping this trait.")
        return None
        
    df['label'] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)
    df_processed = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    
    # For GCN, we use the entire dataset as one graph
    dataset = Dataset.from_pandas(df_processed)
    print("Data preparation complete.")
    return dataset

## ---------------------------------------------------
## --- Tokenizer (loaded once) ---
## ---------------------------------------------------
print("Loading tokenizer...")
tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_LLAMA, token=my_token)
if tokenizer_llama.pad_token is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token

## ---------------------------------------------------
## --- Main Execution Loop ---
## ---------------------------------------------------
for target_trait in ALL_TARGET_COLUMNS:
    print("\n" + "="*80)
    print(f" PROCESSING TRAIT: {target_trait} ".center(80, "="))
    print("="*80)

    output_dir_trait = os.path.join(BASE_OUTPUT_DIR, f"{target_trait}")
    full_dataset = load_and_prepare_data(DATA_FILE, TEXT_COLUMN, target_trait)
    if full_dataset is None:
        continue

    # --- Graph Construction ---
    num_nodes = len(full_dataset)
    # Create a fully connected graph (every node connected to every other node)
    # Keep these tensors on the CPU. The Trainer will move them to the GPU.
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    graph_labels = torch.tensor(full_dataset['label'], dtype=torch.long)
    print(f"Constructed graph with {num_nodes} nodes and {edge_index.shape[1]} edges.")

    # --- Model Initialization ---
    model = LlamaForGraphClassification.from_pretrained(
        MODEL_CHECKPOINT_LLAMA,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=my_token,
    )
    model.config.pad_token_id = tokenizer_llama.pad_token_id

    # Apply LoRA to the Llama backbone
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # --- Tokenize all nodes ---
    def tokenize_function(examples):
        return tokenizer_llama(examples["text"], truncation=True, max_length=128, padding="max_length")
    
    # We tokenize the whole dataset but will only use it for passing to the model
    tokenized_inputs = tokenize_function(full_dataset)
    # The "dataset" for the trainer is just a dummy to control the number of steps
    train_dataset = Dataset.from_dict({'dummy': range(len(full_dataset))})


    # --- Custom Data Collator ---
    # This collator adds the static graph data to every batch.
    class GraphDataCollator:
        def __call__(self, features):
            # The 'features' are just dummy indices, we ignore them
            # Instead, we return the pre-tokenized inputs for the whole graph
            batch = {
                'input_ids': torch.tensor(tokenized_inputs['input_ids']),
                'attention_mask': torch.tensor(tokenized_inputs['attention_mask']),
                'edge_index': edge_index,
                'graph_labels': graph_labels
            }
            return batch

    # --- Trainer Setup ---
    trainer = GCNTrainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=output_dir_trait,
            num_train_epochs=10, # GCNs may benefit from more epochs
            per_device_train_batch_size=1, # We process the whole graph, so batch size is 1
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            learning_rate=5e-5, # GCNs can be sensitive to learning rate
            logging_steps=1,
            save_strategy="epoch",
            eval_strategy="no", # Evaluation would require a separate validation graph
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        data_collator=GraphDataCollator(),
    )

    print(f"\nStarting GCN-based fine-tuning for {target_trait}...")
    trainer.train()

    print(f"Saving final model adapter and tokenizer to {output_dir_trait}")
    peft_model.save_pretrained(output_dir_trait)
    tokenizer_llama.save_pretrained(output_dir_trait)

print("\n" + "="*80)
print(" SCRIPT FINISHED: ALL TRAITS PROCESSED ".center(80, "="))
print("="*80)
