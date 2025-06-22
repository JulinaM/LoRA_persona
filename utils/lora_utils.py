import json, os, torch
from sklearn.metrics import accuracy_score, f1_score
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
    # MODEL_CHECKPOINT = "mistralai/Mistral-7B-v0.1"
    # MODEL_CHECKPOINT = "tiiuae/falcon-7b"
    BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt/llama_lora_class_balanced" 
    TOKEN_MAX_LEN = 128
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
        learning_rate=3e-4,
        # logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        metric_for_best_model="accuracy", 
        greater_is_better=True,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        group_by_length=True,
        save_total_limit=1,
    )


def load_and_prepare_all_data():
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
    
    trainval1, test1_df = train_test_split(df1, test_size=0.1, random_state=42)
    trainval2, test2_df = train_test_split(df2, test_size=0.1, random_state=42)
    trainval_df = pd.concat([trainval1, trainval2], ignore_index=True)
    return trainval1, test1_df, test2_df


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def create_out_dir(target_trait):
    output_dir_trait = os.path.join(Config.BASE_OUTPUT_DIR, target_trait)
    os.makedirs(output_dir_trait, exist_ok=True)
    return output_dir_trait
