import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(MultiModalFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ModalityAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ModalityAlignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, text_features, image_features, tabular_features):
        # Normalize feature vectors
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)
        tabular_features = F.normalize(tabular_features, p=2, dim=1)

        # Compute similarity matrices
        text_image_sim = torch.matmul(text_features, image_features.t()) / self.temperature
        text_tabular_sim = torch.matmul(text_features, tabular_features.t()) / self.temperature
        image_tabular_sim = torch.matmul(image_features, tabular_features.t()) / self.temperature

        # Compute alignment losses
        text_image_loss = F.cross_entropy(text_image_sim, torch.arange(text_features.size(0)).to(text_features.device))
        text_tabular_loss = F.cross_entropy(text_tabular_sim, torch.arange(text_features.size(0)).to(text_features.device))
        image_tabular_loss = F.cross_entropy(image_tabular_sim, torch.arange(image_features.size(0)).to(image_features.device))

        return text_image_loss + text_tabular_loss + image_tabular_loss

class MultiModalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, temperature=0.07, lambda_alignment=0.1):
        super(MultiModalLoss, self).__init__()
        self.focal_loss = MultiModalFocalLoss(alpha, gamma)
        self.alignment_loss = ModalityAlignmentLoss(temperature)
        self.lambda_alignment = lambda_alignment

    def forward(self, outputs, targets, text_features, image_features, tabular_features):
        focal_loss = self.focal_loss(outputs, targets)
        alignment_loss = self.alignment_loss(text_features, image_features, tabular_features)
        total_loss = focal_loss + self.lambda_alignment * alignment_loss
        return total_loss, focal_loss, alignment_loss

# Usage example
if __name__ == "__main__":
    num_classes = 10
    batch_size = 32
    feature_dim = 128

    criterion = MultiModalLoss(num_classes)
    
    # Dummy data
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    text_features = torch.randn(batch_size, feature_dim)
    image_features = torch.randn(batch_size, feature_dim)
    tabular_features = torch.randn(batch_size, feature_dim)

    total_loss, focal_loss, alignment_loss = criterion(outputs, targets, text_features, image_features, tabular_features)
    # print(f"Total Loss: {total_loss.item():.4f}")
    # print(f"Focal Loss: {focal_loss.item():.4f}")
    # print(f"Alignment Loss: {alignment_loss.item():.4f}")