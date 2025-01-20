import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs=10, lr=1e-5, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", unit="batch")
        
        for batch in train_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
        
        train_accuracy = correct_preds / total_preds
        train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct_preds = 0
        val_total_preds = 0
        
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", unit="batch")
        
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct_preds += (preds == labels).sum().item()
                val_total_preds += labels.size(0)
        
        val_accuracy = val_correct_preds / val_total_preds
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), "best_multimodal_model.pth")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping!")
                break
