import os, gc, json, sys
import pandas as pd
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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, load_and_prepare_all_data, compute_mtl_metrics, create_out_dir

class LoRA_config:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        bias="none",
        task_type="SEQ_CLS"
    )
    
    training_args = TrainingArguments(
        output_dir=Config.BASE_OUTPUT_DIR+'/llama_lora_mtl/',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_strategy="epoch", #logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        group_by_length=True,
        save_total_limit=1,
    )

def main():
    MODEL_CHECKPOINT = Config.MODEL_CHECKPOINT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using Model: {MODEL_CHECKPOINT}")
    print('Using args', LoRA_config.training_args)

    print("Loading tokenizer...")
    tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, token=Config.get_hf_token())
    if tokenizer_llama.pad_token is None:
        tokenizer_llama.pad_token = tokenizer_llama.eos_token if tokenizer_llama.pad_token is None else tokenizer_llama.pad_token

    trainval_df, test1_df, test2_df = load_and_prepare_all_data(0.15)
    train_df, val_df = train_test_split(trainval_df, test_size=0.15, random_state=42)
    print(f"\nData split into {len(train_df)} train, {len(val_df)} validation, {len(test1_df)} test1, and {len(test2_df)} test2 samples.")

    raw_dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test1': Dataset.from_pandas(test1_df),
        'test2': Dataset.from_pandas(test2_df)
    })

    def preprocess_function(examples):
        tokenized_inputs = tokenizer_llama(examples['text'], truncation=True, max_length=128)
        labels = np.array([examples[col] for col in Config.ALL_TARGET_COLUMNS])
        tokenized_inputs['labels'] = labels.T.astype(np.float32).tolist()
        return tokenized_inputs
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(Config.ALL_TARGET_COLUMNS),
        problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer_llama.pad_token_id
    model_lora = get_peft_model(model, LoRA_config.lora_config)
    model_lora.print_trainable_parameters()

    trainer = Trainer(
        model=model_lora,
        args=LoRA_config.training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_llama),
        compute_metrics=compute_mtl_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n--- Starting Multi-Task Learning Training ---")
    trainer.train()

    for split_key, split_name in {'test1': "mypersonality", 'test2': "Essay"}.items():
        print(f"\n--- Evaluating on final hold-out test set: {split_name} ---")
        
        prediction_output = trainer.predict(tokenized_dataset[split_key])
        logits = prediction_output.predictions
        true_labels = prediction_output.label_ids

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        binary_preds = (probs >= 0.5).int().numpy()

        f1_micro_overall = f1_score(true_labels, binary_preds, average='micro')
        accuracy_overall = accuracy_score(true_labels, binary_preds) # This is subset accuracy

        print("\n" + "-"*50)
        print(f"Final Test Set '{split_name}' Evaluation Results (Overall):")
        print(f"  Subset Accuracy (Exact Match): {accuracy_overall:.4f}")
        print(f"  F1 Score (micro): {f1_micro_overall:.4f}")
        
        # --- Per-Trait Metrics ---
        print("\nPer-Trait Test Results:")
        for i, trait in enumerate(Config.ALL_TARGET_COLUMNS):
            trait_labels = true_labels[:, i]
            trait_preds = binary_preds[:, i]
            
            trait_acc = accuracy_score(trait_labels, trait_preds)
            trait_f1 = f1_score(trait_labels, trait_preds, average='binary')
            
            print(f"  --- {trait} ---")
            print(f"    Accuracy: {trait_acc:.4f}")
            print(f"    F1 Score: {trait_f1:.4f}")
        print("-"*50)

    print("\n SCRIPT FINISHED ".center(80, "="))
    del model, model_lora, trainer
    gc.collect()
torch.cuda.empty_cache()


if __name__ == "__main__":
    main()