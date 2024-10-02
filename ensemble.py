import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_modal_fusion_model import MultiModalFusionModel

class EnsembleFusionModel(nn.Module):
    def __init__(self, num_classes, tabular_input_dim, num_models=3):
        super(EnsembleFusionModel, self).__init__()
        self.models = nn.ModuleList([
            MultiModalFusionModel(num_classes, tabular_input_dim)
            for _ in range(num_models)
        ])
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, text_ids, text_mask, images, tabular_data):
        outputs = [model(text_ids, text_mask, images, tabular_data) for model in self.models]
        weighted_outputs = torch.stack([w * out for w, out in zip(self.weights, outputs)])
        return torch.sum(weighted_outputs, dim=0)

class DiversityLoss(nn.Module):
    def __init__(self, lambda_div=0.1):
        super(DiversityLoss, self).__init__()
        self.lambda_div = lambda_div
    
    def forward(self, outputs):
        batch_size = outputs[0].size(0)
        num_models = len(outputs)
        
        mean_output = torch.mean(torch.stack(outputs), dim=0)
        diversity_loss = 0
        
        for output in outputs:
            diversity_loss += F.mse_loss(output, mean_output)
        
        return self.lambda_div * diversity_loss / (num_models * batch_size)

def train_ensemble(ensemble_model, train_loader, val_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    diversity_criterion = DiversityLoss()
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loss = 0
        
        for batch in train_loader:
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            images = batch['image'].to(device)
            tabular_data = batch['tabular_data'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            individual_outputs = [model(text_ids, text_mask, images, tabular_data) for model in ensemble_model.models]
            ensemble_output = ensemble_model(text_ids, text_mask, images, tabular_data)
            
            loss = criterion(ensemble_output, labels)
            diversity_loss = diversity_criterion(individual_outputs)
            total_loss = loss + diversity_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        ensemble_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(device)
                text_mask = batch['text_mask'].to(device)
                images = batch['image'].to(device)
                tabular_data = batch['tabular_data'].to(device)
                labels = batch['label'].to(device)
                
                outputs = ensemble_model(text_ids, text_mask, images, tabular_data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step()
    
    return ensemble_model

# Usage example
if __name__ == "__main__":
    num_classes = 10
    tabular_input_dim = 20
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    
    ensemble_model = EnsembleFusionModel(num_classes, tabular_input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model.to(device)
    
    # Assume you have created train_loader and val_loader
    trained_ensemble = train_ensemble(ensemble_model, train_loader, val_loader, num_epochs, learning_rate, device)
    
    # Save the trained ensemble model
    torch.save(trained_ensemble.state_dict(), "trained_ensemble_model.pth")