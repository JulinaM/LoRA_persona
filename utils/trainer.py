import torch 
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
import numpy as np

class Trainer:
    # max_grad_norm = 1.0
    def train_one_epoch(model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device) )
            labels = batch['labels'].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)


    def val_one_epoch(model, data_loader, criterion, device):
        model.eval()
        all_val_preds, all_val_labels = [], []
        v_total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                labels = batch['labels'].to(device)
                logits = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device) )
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
    

    def find_optimal_thresholds(model, val_loader, device, trait_names):
        """Returns a dict mapping trait names to optimal thresholds."""
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device) )
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_labels.append(batch['labels'].cpu())
        
        all_probs = torch.cat(all_probs).numpy()  # Shape: [n_samples, n_traits]
        all_labels = torch.cat(all_labels).numpy()
        
        optimal_thresholds = {}
        for i, trait in enumerate(trait_names):
            precision, recall, thresholds = precision_recall_curve(
                all_labels[:, i], all_probs[:, i]
            )
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
            optimal_idx = np.argmax(f1_scores)
            optimal_thresholds[trait] = float(thresholds[optimal_idx])
        return optimal_thresholds
    
    
    def evaluate(model, data_loader, device, trait_names, split_name, optimal_thresholds=None):
        model.eval()
        all_preds, all_labels = [], []        
        with torch.no_grad():
            for batch in data_loader:
                logits = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device) )
                if optimal_thresholds:
                    probs = torch.sigmoid(logits).cpu() 
                    preds = torch.zeros_like(probs)
                    for i, trait in enumerate(trait_names): 
                        preds[:, i] = (probs[:, i] >= optimal_thresholds[trait]).int()
                else:
                    preds = (torch.sigmoid(logits) > 0.5).int()
                all_preds.append(preds.cpu())
                all_labels.append(batch['labels'].to(device).cpu())
            
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            f1_micro = f1_score(all_labels, all_preds, average='micro')
            accuracy = accuracy_score(all_labels, all_preds)

            print("\n" + "-"*50)
            print(f"Final Test Set '{split_name}' Evaluation Results (Overall):")
            print(f"  Subset Accuracy (Exact Match): {accuracy:.4f}")
            print(f"  F1 Score (micro): {f1_micro:.4f}")
            print(f'Using {optimal_thresholds}')
            
            print("\nPer-Trait Test Results:")
            for i, trait in enumerate(trait_names):
                trait_acc = accuracy_score(all_labels[:, i], all_preds[:, i])
                trait_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary')
                
                print(f"  --- {trait} ---")
                print(f"    Accuracy: {trait_acc:.4f}")
                print(f"    F1 Score: {trait_f1:.4f}")
            print("-"*50)
    
