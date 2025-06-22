import gc, os, torch, sys
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
from peft import get_peft_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, LoRA_config, load_and_prepare_all_data, compute_metrics, create_out_dir


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

    # print(f"\nBalancing training data for class labels in trait: {target_trait}")
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

    model_llama = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_llama.config.pad_token_id = tokenizer_llama.pad_token_id
    model_llama_lora = get_peft_model(model_llama,  LoRA_config.lora_config)
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