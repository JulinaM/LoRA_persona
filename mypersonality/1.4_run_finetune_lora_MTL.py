import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch
import gc
import json # <<< MODIFIED: Added json import


## ---------------------------------------------------
## --- Configuration ---
## ---------------------------------------------------
DATA_URL_MYPERSONALITY = '/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv'
DATA_URL_ESSAY = '/users/PGS0218/julina/projects/LoRA_persona/data/essay.csv'
ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
MODEL_CHECKPOINT = "meta-llama/Meta-Llama-3-8B"
BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt/llama_lora_mtl" # New dir for MTL
try:
    with open('/users/PGS0218/julina/projects/LoRA_persona/data/keys.json', 'r') as f:
        keys = json.load(f)
        my_token = keys['hf_read']
except (FileNotFoundError, KeyError) as e:
    print(f"Error reading token: {e}. Please ensure '../data/keys.json' exists and contains the 'hf_read' key.")
    my_token = None 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"Using Model: {MODEL_CHECKPOINT}")

## ---------------------------------------------------
## --- Data Loading and Metrics Functions ---
## ---------------------------------------------------
def load_and_prepare_all_data():
    print("--- Loading and Preparing All Datasets ---")
    def load_mypersonality_data():
        df = pd.read_csv(DATA_URL_MYPERSONALITY, encoding='Windows-1252')
        df = df.rename(columns={'STATUS': 'text'})
        df['text'] = df['text'].fillna('')
        df = df[['text'] + ALL_TARGET_COLUMNS]
        for col in ALL_TARGET_COLUMNS:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)
        return df

    def load_essay_data():
        df = pd.read_csv(DATA_URL_ESSAY, encoding='utf-8')
        df['text'] = df['text'].fillna('')
        return df

    df1 = load_mypersonality_data()
    df2 = load_essay_data()
    
    trainval1, test1_df = train_test_split(df1, test_size=0.1, random_state=42)
    trainval2, test2_df = train_test_split(df2, test_size=0.1, random_state=42)
    trainval_df = pd.concat([trainval1, trainval2], ignore_index=True)
    return trainval1, test1_df, test2_df


def compute_mtl_metrics(p):
    """
    Computes metrics for multi-label classification.
    p: EvalPrediction object containing predictions and label_ids.
    """
    logits = p.predictions
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

print("Loading tokenizer...")
tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, token=my_token)
if tokenizer_llama.pad_token is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token
    print("Tokenizer `pad_token` set to `eos_token`.")

# Load the single dataset, prepped for MTL
trainval_df, test1_df, test2_df = load_and_prepare_all_data()
train_df, val_df = train_test_split(trainval_df, test_size=0.1, random_state=42)
print(f"\nData split into {len(train_df)} train, {len(val_df)} validation, and {len(test1_df)} test samples.")

tokenized_dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df),
    'test1': Dataset.from_pandas(test1_df),
    'test2': Dataset.from_pandas(test2_df)

})

# Since MTL is a more complex task, we might need a slightly larger adapter capacity.
lora_config = LoraConfig(
    r=32, 
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(ALL_TARGET_COLUMNS),  # 5 labels for 5 traits
    problem_type="multi_label_classification", # CRITICAL FOR MTL
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer_llama.pad_token_id

model_lora = get_peft_model(model, lora_config)
model_lora.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=BASE_OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # For MTL, micro F1 or ROC AUC are good metrics to track
    metric_for_best_model="f1_micro",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    group_by_length=True,
    save_total_limit=1,
)

trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer_llama),
    compute_metrics=compute_mtl_metrics,
)

## ---------------------------------------------------
## --- Train and Evaluate ---
## ---------------------------------------------------
print("\n--- Starting Multi-Task Learning Training ---")
trainer.train()

for k,v in {'test1': "mypersonality", 'test2': "Essay"}.items():
    print(f"\n--- Evaluating on final hold-out test set {v} ---")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["k"])
    print("\n" + "-"*50)
    print(f"Final Test Set {v} Evaluation Results (MTL):")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print("-" * 50)


print("\n SCRIPT FINISHED ".center(80, "="))
del model, model_lora, trainer
gc.collect()
torch.cuda.empty_cache()
