import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from multi_modal_fusion_model import MultiModalFusionModel
from multi_modal_dataset import MultiModalDataset

def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            images = batch['image'].to(device)
            tabular_data = batch['tabular_data'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(text_ids, text_mask, images, tabular_data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(device)
                text_mask = batch['text_mask'].to(device)
                images = batch['image'].to(device)
                tabular_data = batch['tabular_data'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(text_ids, text_mask, images, tabular_data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return model

if __name__ == "__main__":
    # Hyperparameters
    num_classes = 10
    tabular_input_dim = 20
    batch_size = 32
    num_epochs = 10
    learning_rate = 2e-5
    
    # Set up data
    csv_file = ''
    img_dir = ''
    text_column = 'text_description'
    img_column = 'image_filename'
    tabular_columns = []
    label_column = 'target'
    
    full_dataset = MultiModalDataset(csv_file, img_dir, text_column, img_column, tabular_columns, label_column)
    
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFusionModel(num_classes, tabular_input_dim).to(device)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)