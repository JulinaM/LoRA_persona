import torch 
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
import numpy as np

class Trainer:
    def train_one_epoch(model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def val_one_epoch(model, data_loader, criterion, device):
        model.eval()
        all_val_preds, all_val_labels = [], []
        v_total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                v_loss = criterion(logits, labels)
                v_total_loss += v_loss.item()
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())
        avg_val_loss = v_total_loss / len(data_loader)

        val_preds = torch.cat(all_val_preds).numpy()
        val_labels = torch.cat(all_val_labels).numpy()
        val_f1 = f1_score(val_labels, val_preds, average='micro')
        val_acc = accuracy_score(val_labels, val_preds)
        return {'loss': avg_val_loss, 'f1_macro': val_f1, "accuracy": val_acc}
    

    def find_optimal_thresholds(model, val_loader, device):
        """Find optimal sigmoid thresholds per trait using validation set."""
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
        all_probs = torch.cat(all_probs).numpy()  # Shape: [n_val_samples, n_traits]
        all_labels = torch.cat(all_labels).numpy()
        
        optimal_thresholds = []
        for i in range(all_labels.shape[1]):  # Loop per trait
            precision, recall, thresholds = precision_recall_curve(
                all_labels[:, i], 
                all_probs[:, i]
            )
            # Calculate F1 for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
            best_threshold = thresholds[np.argmax(f1_scores)]
            optimal_thresholds.append(best_threshold)
        return optimal_thresholds
    
    
    def evaluate(model, data_loader, device, labels, split_name, optimal_thresholds=None):
        model.eval()
        all_test_preds, all_test_labels = [], []
        with torch.no_grad():

            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()  
                if optimal_thresholds:
                    preds = np.zeros_like(probs)
                    for i, threshold in enumerate(optimal_thresholds):
                        preds[:, i] = (probs[:, i] >= threshold).astype(int)  
                    preds = torch.from_numpy(preds)
                else:
                    preds = (torch.sigmoid(logits) > 0.5).int()
                all_test_preds.append(preds.cpu())
                all_test_labels.append(labels.cpu())
            
            test_preds = torch.cat(all_test_preds).numpy()
            true_labels = torch.cat(all_test_labels).numpy()

            f1_micro = f1_score(true_labels, test_preds, average='micro')
            accuracy = accuracy_score(true_labels, test_preds)

            print("\n" + "-"*50)
            print(f"Final Test Set '{split_name}' Evaluation Results (Overall):")
            print(f"  Subset Accuracy (Exact Match): {accuracy:.4f}")
            print(f"  F1 Score (micro): {f1_micro:.4f}")
            
            print("\nPer-Trait Test Results:")
            for i, trait in enumerate(labels):
                trait_acc = accuracy_score(true_labels[:, i], test_preds[:, i])
                trait_f1 = f1_score(true_labels[:, i], test_preds[:, i], average='binary')
                
                print(f"  --- {trait} ---")
                print(f"    Accuracy: {trait_acc:.4f}")
                print(f"    F1 Score: {trait_f1:.4f}")
            print("-"*50)
    
