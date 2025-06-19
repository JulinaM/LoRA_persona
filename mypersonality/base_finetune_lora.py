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
from sklearn.utils.class_weight import compute_class_weight
import torch
import getpass

# (The CustomTrainer for handling class imbalance correctly remains the same)
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train_dataset is not None:
            train_labels = self.train_dataset["label"]
            if len(np.unique(train_labels)) > 1:
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
                self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)
                print(f"Using class weights to handle class imbalance: {self.class_weights}")
            else:
                self.class_weights = None
                print("Only one class present in training data. Not using class weights.")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Use class weights only if they have been computed
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def run_training_pipeline(dataset_name: str, data_path: str, text_column_name: str, hf_token: str):
    """
    A full pipeline to load one dataset, train, and evaluate.
    """
    print("\n" + "#"*100)
    print(f"  STARTING PIPELINE FOR: {dataset_name.upper()} DATASET  ".center(100, "#"))
    print("#"*100 + "\n")

    # --- 1. Load Data ---
    print(f"--- Loading {dataset_name} data from {data_path} ---")
    if dataset_name == "mypersonality":
        df = pd.read_csv(data_path, encoding='Windows-1252')
    else:
        df = pd.read_csv(data_path, encoding='utf-8')
        
    df = df.rename(columns={text_column_name: 'text'})
    df['text'] = df['text'].fillna('')
    df = df[['text'] + ALL_TARGET_COLUMNS]
    for col in ALL_TARGET_COLUMNS:
        df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)

    # --- 2. Initialize Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_LLAMA, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Loop through each personality trait ---
    for target_trait in ALL_TARGET_COLUMNS:
        print("\n" + "="*80)
        print(f" PROCESSING TRAIT: {target_trait} on {dataset_name} data ".center(80, "="))
        print("="*80)
        
        # Check if there are enough samples for both classes
        if df[target_trait].nunique() < 2:
            print(f"Skipping trait '{target_trait}' for {dataset_name} due to only one class being present.")
            continue

        output_dir_trait = os.path.join(BASE_OUTPUT_DIR, dataset_name, f"{target_trait}")
        os.makedirs(output_dir_trait, exist_ok=True)

        # --- Data Splitting and Preparation ---
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_trait])
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df[target_trait])

        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })

        trait_dataset_dict = dataset_dict.rename_column(target_trait, "label")
        columns_to_remove = [col for col in ALL_TARGET_COLUMNS if col != "label"] + ["text"]

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=1024)

        tokenized_dataset = trait_dataset_dict.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
        
        # --- Model and LoRA Configuration ---
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT_LLAMA, num_labels=2, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # --- Trainer Setup and Execution ---
        training_args = TrainingArguments(
            output_dir=output_dir_trait, num_train_epochs=4, per_device_train_batch_size=8,
            gradient_accumulation_steps=4, learning_rate=3e-5, logging_steps=10,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="f1", weight_decay=0.01, lr_scheduler_type="cosine",
            group_by_length=True,
        )

        trainer = CustomTrainer(
            model=model, args=training_args,
            train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["validation"],
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        
        # --- Final Evaluation ---
        print(f"\n--- Final Evaluation on {dataset_name} Test Set for {target_trait} ---")
        test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    DATA_URL_MYPERSONALITY = '/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv'
    DATA_URL_ESSAY = '/users/PGS0218/julina/projects/LoRA_persona/data/essay.csv'
    ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    MODEL_CHECKPOINT_LLAMA = "meta-llama/Meta-Llama-3-8B"
    BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt_llama_lora_isolated"
    MY_TOKEN = ""

    # --- Pipeline Execution ---
    # RUN 1: Train and evaluate ONLY on Essay data
    run_training_pipeline(
        dataset_name="essay",
        data_path=DATA_URL_ESSAY,
        text_column_name="TEXT",
        hf_token=MY_TOKEN
    )

    # RUN 2: Train and evaluate ONLY on Mypersonality (Facebook) data
    run_training_pipeline(
        dataset_name="mypersonality",
        data_path=DATA_URL_MYPERSONALITY,
        text_column_name="STATUS",
        hf_token=MY_TOKEN
    )

    print("\n" + "#"*100)
    print("  ALL PIPELINES FINISHED  ".center(100, "#"))
    print("#"*100 + "\n")