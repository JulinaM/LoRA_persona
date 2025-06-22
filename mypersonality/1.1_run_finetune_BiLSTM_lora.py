import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
import gc
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutput

from lora_utils import Config, LoRA_config, load_and_prepare_all_data, compute_metrics, create_out_dir

class ClassifierHead(LlamaPreTrainedModel):
    def __init__(self, config, mlp_hidden_size=256, dropout_rate=0.3):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, self.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs,)
        
        hidden_states = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Pass the pooled sentence embedding to our classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"Using ModelL: {Config.MODEL_CHECKPOINT}")
print("Loading tokenizer...")
tokenizer_llama = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT, token=Config.get_hf_token())
if tokenizer_llama.pad_token is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token

trainval_df, test1_df, test2_df = load_and_prepare_all_data()
for target_trait in Config.ALL_TARGET_COLUMNS:
    print("\n" + "="*80)
    print(f" PROCESSING TRAIT: {target_trait} ".center(80, "="))
    print("="*80)

    print(f"\nBalancing training data for class labels in trait: {target_trait}")
    train_df, val_df = train_test_split(trainval_df, test_size=0.1, random_state=42, stratify=trainval_df[target_trait])
    # X_train = train_df.drop(columns=Config.ALL_TARGET_COLUMNS)
    # y_train = train_df[target_trait]
    # print(f"Original training distribution for {target_trait}: \n{y_train.value_counts(normalize=True)}")
    
    # ros = RandomOverSampler(random_state=42) # Use RandomOverSampler to balance the TRAINING data
    # X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    # train_df_balanced = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    # print(f"Balanced training distribution for {target_trait}: \n{train_df_balanced[target_trait].value_counts(normalize=True)}")
    
    base_dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df), # Validation set is NOT resampled
        'test1': Dataset.from_pandas(test1_df),   # Test sets are NOT resampled
        'test2': Dataset.from_pandas(test2_df)
    })
    # --- End of New Balancing Section ---

    print(f"\nTokenizing data...")
    trait_dataset_dict = base_dataset_dict.rename_column(target_trait, "label")
    def tokenize_function(examples):
        return tokenizer_llama(examples["text"], truncation=True, max_length=Config.TOKEN_MAX_LEN)
    tokenized_dataset = trait_dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"])

    model_llama = ClassifierHead.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=Config.get_hf_token,
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