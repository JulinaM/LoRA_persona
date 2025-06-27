import json, os, torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from peft import  LoraConfig, TaskType
import pandas as pd
import numpy as np
from transformers import TrainingArguments

class Config:
    DATA_URL_MYPERSONALITY = '/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv'
    DATA_URL_ESSAY = '/users/PGS0218/julina/projects/LoRA_persona/data/essay.csv'
    ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    MODEL_CHECKPOINT = "meta-llama/Meta-Llama-3-8B"
    MISTRAL_CHECKPOINT = "mistralai/Mistral-7B-v0.1"
    FALCON_CHECKPOINT = "tiiuae/falcon-7b"
    BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt/" 
    TOKEN_MAX_LEN = 128
    MODEL_CHECKPOINT_ROBERTA = "bert-base-uncased"
    TEXT_COLUMN = "STATUS"

    def get_hf_token():
        try:
            with open('/users/PGS0218/julina/projects/LoRA_persona/data/keys.json', 'r') as f:
                keys = json.load(f)
                my_token = keys['hf_read']
        except (FileNotFoundError, KeyError) as e:
            print(f"Error reading token: {e}. Please ensure '../data/keys.json' exists and contains the 'hf_read' key.")
            my_token = None 
        return my_token
    
class LoRA_config:
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1, #0.1
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        )   
    training_args_template = TrainingArguments (
        # logging_dir=f"{BASE_OUTPUT_DIR}/logs",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_strategy = "epoch",  #logging_steps=25,
        warmup_steps=200, 
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        metric_for_best_model="f1", 
        greater_is_better=True,
        weight_decay=0.05,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        group_by_length=True,
        save_total_limit=1,
    )


def load_and_prepare_all_data(split_ratio=0.1, both=False):
    print("--- Loading and Preparing All Datasets ---")
    def load_mypersonality_data():
        df = pd.read_csv(Config.DATA_URL_MYPERSONALITY, encoding='Windows-1252')
        df = df.rename(columns={'STATUS': 'text'})
        df['text'] = df['text'].fillna('')
        df = df[['text'] + Config.ALL_TARGET_COLUMNS]
        for col in Config.ALL_TARGET_COLUMNS:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)
        return df

    def load_essay_data():
        df = pd.read_csv(Config.DATA_URL_ESSAY, encoding='utf-8')
        df['text'] = df['text'].fillna('')
        return df

    df1 = load_mypersonality_data()
    df2 = load_essay_data()
    
    train1, test1_df = train_test_split(df1, test_size=split_ratio*2, random_state=42)
    val1, test1_df = train_test_split(test1_df, test_size=0.5, random_state=42)

    train2, test2_df = train_test_split(df2, test_size=split_ratio*2, random_state=42)
    val2, test2_df = train_test_split(test2_df, test_size=0.5, random_state=42)

    if both:
        print("Merging fb and essay dataset:")
        train = pd.concat([train1, train2], ignore_index=True)
        val = pd.concat([val1, val2], ignore_index=True)
        return train, val, test1_df, test2_df
    
    return train1, val1, test1_df, test2_df


def compute_pos_weight(train_df, device):
    labels_df = train_df[Config.ALL_TARGET_COLUMNS]
    pos_counts = labels_df.sum()
    neg_counts = len(labels_df) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8) # Clamp to avoid division by zero if a class has 0 positive samples
    pos_weight_tensor = torch.tensor(pos_weight.values, dtype=torch.float).to(device)
    print("\nCalculated positive class weights for weighted loss:")
    print(pos_weight)
    return pos_weight_tensor


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def compute_mtl_metrics(p):
    """Computes overall micro-F1 and ROC-AUC for multi-label evaluation."""
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1

    f1_micro = f1_score(labels, predictions, average='micro')
    roc_auc = roc_auc_score(labels, probs, average='micro')
    accuracy = accuracy_score(labels, predictions)
    metrics = {
        'f1_micro': f1_micro,
        'roc_auc_micro': roc_auc,
        'accuracy': accuracy
    }
    return metrics

def create_unique_dir(task_name):
    from datetime import datetime
    output_dir = os.path.join(Config.BASE_OUTPUT_DIR + '/' + task_name, datetime.now().strftime("%Y%m%d_%H%M%S")[-6:])
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
