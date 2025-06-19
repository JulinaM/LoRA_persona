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
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from imblearn.over_sampling import RandomOverSampler
import gc

## ---------------------------------------------------
## --- Configuration ---
## ---------------------------------------------------
DATA_URL_MYPERSONALITY = '/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv'
DATA_URL_ESSAY = '/users/PGS0218/julina/projects/LoRA_persona/data/essay.csv'
ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
MODEL_CHECKPOINT_LLAMA = "meta-llama/Meta-Llama-3-8B"
BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt/llama_lora_class_balanced" 
my_token = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

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


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

print("Loading tokenizer...")
tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_LLAMA, token=my_token)
if tokenizer_llama.pad_token is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token

# Load the raw dataframes first
trainval_df, test1_df, test2_df = load_and_prepare_all_data()

# <<< BEST PRACTICE: Define static configs once
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.07,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
training_args_template = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    metric_for_best_model="f1",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    group_by_length=True,
    logging_dir=f"{BASE_OUTPUT_DIR}/logs",
    report_to="tensorboard",
    save_total_limit=1, # Only save the best checkpoint
)

## ---------------------------------------------------
## --- Main Execution Loop for All Traits ---
## ---------------------------------------------------
for target_trait in ['cOPN', 'cEXT']:
    print("\n" + "="*80)
    print(f" PROCESSING TRAIT: {target_trait} ".center(80, "="))
    print("="*80)

    print(f"\nBalancing training data for class labels in trait: {target_trait}")
    train_df, val_df = train_test_split(trainval_df, test_size=0.1, random_state=42, stratify=trainval_df[target_trait])
    X_train = train_df.drop(columns=ALL_TARGET_COLUMNS)
    y_train = train_df[target_trait]
    print(f"Original training distribution for {target_trait}: \n{y_train.value_counts(normalize=True)}")
    
    ros = RandomOverSampler(random_state=42) # Use RandomOverSampler to balance the TRAINING data
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    train_df_balanced = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    print(f"Balanced training distribution for {target_trait}: \n{train_df_balanced[target_trait].value_counts(normalize=True)}")
    
    base_dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df_balanced),
        'validation': Dataset.from_pandas(val_df), # Validation set is NOT resampled
        'test1': Dataset.from_pandas(test1_df),   # Test sets are NOT resampled
        'test2': Dataset.from_pandas(test2_df)
    })
    # --- End of New Balancing Section ---

    print(f"\nTokenizing data...")
    trait_dataset_dict = base_dataset_dict.rename_column(target_trait, "label")
    def tokenize_function(examples):
        return tokenizer_llama(examples["text"], truncation=True, max_length=128)
    tokenized_dataset = trait_dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"])

    model_llama = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT_LLAMA,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_llama.config.pad_token_id = tokenizer_llama.pad_token_id
    model_llama_lora = get_peft_model(model_llama, lora_config)
    model_llama_lora.print_trainable_parameters()

    output_dir_trait = os.path.join(BASE_OUTPUT_DIR, target_trait)
    os.makedirs(output_dir_trait, exist_ok=True)
    training_args = training_args_template
    training_args.output_dir = output_dir_trait 

    trainer_llama = Trainer(
        model=model_llama_lora,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_llama),
        compute_metrics=compute_metrics,
    )
    trainer_llama.train()

    for k,v in {'test1': "mypersonality", 'test2': "Essay"}.items():
        print("\n" + "-"*50 , f"f or {target_trait}")
        print(f"Evaluating on Test Set {v} for trait: {target_trait}")
        test_results = trainer_llama.evaluate(eval_dataset=tokenized_dataset[k])
        print(f"\n--- Test Set {v} Evaluation Results ---")
        for key, value in test_results.items():
            print(f" {key}: {value:.4f}")
        print("\n" + "-"*50)
    
    del model_llama, model_llama_lora, trainer_llama
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*80)
print(" SCRIPT FINISHED: ALL TRAITS PROCESSED ".center(80, "="))
print("="*80)