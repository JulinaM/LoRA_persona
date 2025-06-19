# TraitBertGCN: A Conceptual Implementation
# This script provides a full, runnable implementation of a BertGCN model for personality trait prediction,
# based on the likely architecture described by papers like "TraitBertGCN".
# It is designed to be run in a local environment or a similar one with GPU support.

# Step 1: Install necessary libraries
# !pip install transformers torch torch_geometric pandas numpy scikit-learn nltk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch_geometric
from transformers import BertTokenizer, BertModel

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from collections import defaultdict
import re
from tqdm import tqdm

# Download NLTK resources (only needs to be done once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True) 


# Step 2: Configuration Block
# A central place to manage hyperparameters and settings.
class Config:
    # MODIFICATION: Add URL for the second dataset
    DATA_URL_MYPERSONALITY = '/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv'
    DATA_URL_ESSAY = '/users/PGS0218/julina/projects/LoRA_persona/data/essay.csv'
    
    BERT_MODEL_NAME = 'bert-base-uncased'
    GRAPH_WINDOW_SIZE = 5  # Co-occurrence window for building graph edges
    BERT_HIDDEN_DIM = 768
    GCN_HIDDEN_DIM = 200
    FINAL_HIDDEN_DIM = 128
    trait_cols = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    NUM_TRAITS = 5
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 3: Data Loading and Preprocessing
class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text_for_bert(self, text):
        """Minimal cleaning for BERT input."""
        text = re.sub(r'#', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_text_for_graph(self, text):
        """Traditional preprocessing for building the GCN graph."""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
        tokens = word_tokenize(text)
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(word) for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]
        return lemmatized_tokens

# MODIFICATION: Separate loading functions for each dataset
def load_mypersonality_data(preprocessor):
    """Loads and preprocesses the mypersonality dataset."""
    print("Loading mypersonality dataset...")
    df = pd.read_csv(Config.DATA_URL_MYPERSONALITY, encoding='Windows-1252')
    df = df.rename(columns={'STATUS': 'text'})
    # FIX: Fill any missing text values with an empty string to prevent type errors.
    df['text'] = df['text'].fillna('')
    df = df[['text'] + Config.trait_cols]
    for col in Config.trait_cols:     
        df[col] = df[col].apply(lambda x: 1 if x == 'y' else 0)
    df['text_for_bert'] = df['text'].apply(preprocessor.clean_text_for_bert)
    df['text_for_graph'] = df['text'].apply(preprocessor.preprocess_text_for_graph)
    print(f"Mypersonality data loaded. Total samples: {len(df)}")
    return df

def load_essay_data(preprocessor):
    """Loads and preprocesses the essay dataset."""
    print("Loading essay dataset...")
    df = pd.read_csv(Config.DATA_URL_ESSAY)
    # FIX: Fill any missing text values with an empty string for robustness.
    df['text'] = df['text'].fillna('')
    df['text_for_bert'] = df['text'].apply(preprocessor.clean_text_for_bert)
    df['text_for_graph'] = df['text'].apply(preprocessor.preprocess_text_for_graph)
    print(f"Essay data loaded. Total samples: {len(df)}")
    return df[['text', 'text_for_bert', 'text_for_graph'] + Config.trait_cols]

# Step 4: Graph Construction
def build_graph(corpus, window_size):
    """Builds a co-occurrence graph from the corpus."""
    print("Building word co-occurrence graph...")
    word_counts = defaultdict(int)
    for doc in corpus:
        # This loop is now safe because we ensured all text entries are strings.
        for word in doc:
            word_counts[word] += 1
    vocab = sorted(word_counts.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    co_occurrence = defaultdict(int)
    for doc in tqdm(corpus, desc="Processing co-occurrences"):
        for i in range(len(doc)):
            for j in range(i + 1, min(i + window_size, len(doc))):
                w1, w2 = sorted((doc[i], doc[j]))
                if w1 != w2:
                    idx1, idx2 = word_to_idx[w1], word_to_idx[w2]
                    co_occurrence[(idx1, idx2)] += 1
    edge_index = torch.tensor(list(co_occurrence.keys()), dtype=torch.int64).t().contiguous()
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    graph = Data(x=torch.ones(vocab_size, 1), edge_index=edge_index)
    print("Graph construction complete.")
    return graph, word_to_idx, vocab_size

# Step 5: Custom PyTorch Dataset
class PersonalityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=156):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# Step 6: The BertGCN Model Architecture
class BertGCN(nn.Module):
    def __init__(self, bert_model, gcn_hidden_dim, final_hidden_dim, num_traits, vocab_size):
        super(BertGCN, self).__init__()
        self.bert = bert_model
        self.gcn_node_embeddings = nn.Embedding(vocab_size, Config.BERT_HIDDEN_DIM)
        self.gcn1 = GCNConv(Config.BERT_HIDDEN_DIM, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.fusion = nn.Linear(Config.BERT_HIDDEN_DIM + gcn_hidden_dim, final_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(final_hidden_dim, num_traits)

    def forward(self, input_ids, attention_mask, graph_data):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.pooler_output
        node_features = self.gcn_node_embeddings.weight
        node_features = self.gcn1(node_features, graph_data.edge_index)
        node_features = self.relu(node_features)
        node_features = self.gcn2(node_features, graph_data.edge_index)
        graph_embedding = torch.mean(node_features, dim=0).unsqueeze(0).repeat(cls_embedding.size(0), 1)
        combined_embedding = torch.cat([cls_embedding, graph_embedding], dim=1)
        fused_output = self.fusion(combined_embedding)
        fused_output = self.relu(fused_output)
        fused_output = self.dropout(fused_output)
        logits = self.classifier(fused_output)
        return logits

# Step 7: Training and Evaluation Functions
def train_epoch(model, data_loader, graph_data, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    graph_data = graph_data.to(device)
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, graph_data)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, graph_data, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    graph_data = graph_data.to(device)
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask, graph_data)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    all_labels_np, all_preds_np = np.array(all_labels), np.array(all_preds)
    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels_np, all_preds_np, average='micro', zero_division=0)
    return avg_loss, accuracy, f1_macro, f1_micro

# Step 8: Main Execution Block
def main():
    """Main function to run the entire pipeline."""
    preprocessor = DataPreprocessor()
    
    # Load and split both datasets
    df1 = load_mypersonality_data(preprocessor)
    df2 = load_essay_data(preprocessor)
    
    trainval1, test1 = train_test_split(df1, test_size=0.2, random_state=42)
    trainval2, test2 = train_test_split(df2, test_size=0.2, random_state=42)

    # Combine the trainval portions
    trainval_df = pd.concat([trainval1, trainval2], ignore_index=True)
    
    # Split the combined data into final training and validation sets
    train_df, val_df = train_test_split(trainval_df, test_size=0.2, random_state=42)

    print("\n--- Data Split Summary ---")
    print(f"Total training samples: {len(train_df)}")
    print(f"Total validation samples: {len(val_df)}")
    print(f"Test samples (mypersonality): {len(test1)}")
    print(f"Test samples (essays): {len(test2)}")

    # Build graph from the combined training data corpus
    train_corpus = train_df['text_for_graph'].tolist()
    graph, word_to_idx, vocab_size = build_graph(train_corpus, Config.GRAPH_WINDOW_SIZE)
    
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    
    # Create datasets and dataloaders
    train_dataset = PersonalityDataset(texts=train_df['text_for_bert'].values, labels=train_df[Config.trait_cols].values, tokenizer=tokenizer)
    val_dataset = PersonalityDataset(texts=val_df['text_for_bert'].values, labels=val_df[Config.trait_cols].values, tokenizer=tokenizer)
    test1_dataset = PersonalityDataset(texts=test1['text_for_bert'].values, labels=test1[Config.trait_cols].values, tokenizer=tokenizer)
    test2_dataset = PersonalityDataset(texts=test2['text_for_bert'].values, labels=test2[Config.trait_cols].values, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test1_loader = DataLoader(test1_dataset, batch_size=Config.BATCH_SIZE)
    test2_loader = DataLoader(test2_dataset, batch_size=Config.BATCH_SIZE)

    # Initialize model
    print("\nInitializing BertGCN model...")
    bert_base = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
    
    model = BertGCN(bert_model=bert_base, gcn_hidden_dim=Config.GCN_HIDDEN_DIM, final_hidden_dim=Config.FINAL_HIDDEN_DIM, num_traits=Config.NUM_TRAITS, vocab_size=vocab_size)
    model.to(Config.DEVICE)
    print(f"Model moved to {Config.DEVICE}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, graph, loss_fn, optimizer, Config.DEVICE)
        print(f"  Training Loss: {train_loss:.4f}")
        
        # Validation on the combined validation set
        val_loss, val_acc, val_f1_macro, val_f1_micro = eval_model(model, val_loader, graph, loss_fn, Config.DEVICE)
        print(f"  Validation Loss: {val_loss:.4f} | Subset Accuracy: {val_acc:.4f} | F1-Macro: {val_f1_macro:.4f} | F1-Micro: {val_f1_micro:.4f}")

    print("\nTraining complete.")

    # --- Final Evaluation on Separate Test Sets ---
    print("\n--- Final Evaluation on Test Set 1 (mypersonality) ---")
    test1_loss, test1_acc, test1_f1_macro, test1_f1_micro = eval_model(model, test1_loader, graph, loss_fn, Config.DEVICE)
    print(f"  Test Loss: {test1_loss:.4f} | Subset Accuracy: {test1_acc:.4f} | F1-Macro: {test1_f1_macro:.4f} | F1-Micro: {test1_f1_micro:.4f}")

    print("\n--- Final Evaluation on Test Set 2 (essays) ---")
    test2_loss, test2_acc, test2_f1_macro, test2_f1_micro = eval_model(model, test2_loader, graph, loss_fn, Config.DEVICE)
    print(f"  Test Loss: {test2_loss:.4f} | Subset Accuracy: {test2_acc:.4f} | F1-Macro: {test2_f1_macro:.4f} | F1-Micro: {test2_f1_micro:.4f}")

if __name__ == '__main__':
    main()
