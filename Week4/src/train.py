import torch
import csv
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, patience=10, min_delta=0.001):
    best_val = 0
    history = []
    loss_history = []
    epochs_no_improve = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', ncols=100)
        for x,y in train_pbar:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        # Validation with progress bar
        val_acc = evaluate(model, val_loader, device, epoch, epochs)
        history.append(val_acc)
        
        # Check for improvement
        if val_acc > best_val + min_delta:
            best_val = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            best_marker = " 🌟 NEW BEST!"
        else:
            epochs_no_improve += 1
            best_marker = ""
        
        # Warning for potential overfitting
        warning = ""
        if epoch > 0 and loss_history[-1] < loss_history[-2] and val_acc < history[-2]:
            warning = " ⚠️ Possible overfitting!"
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}{best_marker}{warning}")
        
        if epochs_no_improve > 0:
            print(f"  └─ No improvement for {epochs_no_improve}/{patience} epochs")
        
        print(f"{'-'*60}")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n{'!'*60}")
            print(f"Early stopping triggered! No improvement for {patience} epochs.")
            print(f"Best validation accuracy: {best_val:.4f} (Epoch {best_epoch+1})")
            print(f"{'!'*60}\n")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best validation accuracy: {best_val:.4f}")
    print(f"Total epochs trained: {epoch+1}/{epochs}")
    print(f"{'='*60}\n")

    return history, best_val

def evaluate(model, loader, device, epoch=None, total_epochs=None):
    model.eval()
    correct, total = 0, 0
    
    # Add progress bar for validation
    desc = f'Epoch {epoch+1}/{total_epochs} [Val]' if epoch is not None else 'Evaluating'
    val_pbar = tqdm(loader, desc=desc, ncols=100, leave=False)
    
    with torch.no_grad():
        for x,y in val_pbar:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            val_pbar.set_postfix({'acc': f'{correct/total:.4f}'})
    
    return correct / total
