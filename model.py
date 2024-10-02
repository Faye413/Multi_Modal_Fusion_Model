import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", freeze_bert=False):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token representation

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        return self.resnet(images).squeeze(-1).squeeze(-1)

class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(TabularEncoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, tabular_data):
        return self.encoder(tabular_data)

class MultiModalFusionModel(nn.Module):
    def __init__(self, num_classes, tabular_input_dim, fusion_output_dim=512):
        super(MultiModalFusionModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder(tabular_input_dim)
        
        self.text_projection = nn.Linear(768, fusion_output_dim)
        self.image_projection = nn.Linear(2048, fusion_output_dim)
        self.tabular_projection = nn.Linear(128, fusion_output_dim)
        
        self.fusion_layer = nn.MultiheadAttention(embed_dim=fusion_output_dim, num_heads=8)
        self.layer_norm = nn.LayerNorm(fusion_output_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_ids, text_mask, images, tabular_data):
        text_features = self.text_encoder(text_ids, text_mask)
        image_features = self.image_encoder(images)
        tabular_features = self.tabular_encoder(tabular_data)
        
        text_proj = self.text_projection(text_features).unsqueeze(0)
        image_proj = self.image_projection(image_features).unsqueeze(0)
        tabular_proj = self.tabular_projection(tabular_features).unsqueeze(0)
        
        # Concatenate all modalities
        combined_features = torch.cat([text_proj, image_proj, tabular_proj], dim=0)
        
        # Self-attention fusion
        fused_features, _ = self.fusion_layer(combined_features, combined_features, combined_features)
        fused_features = self.layer_norm(fused_features + combined_features)  # Residual connection
        
        # Pool fused features
        pooled_features = torch.mean(fused_features, dim=0)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

# Usage example
if __name__ == "__main__":
    num_classes = 10
    tabular_input_dim = 20
    model = MultiModalFusionModel(num_classes, tabular_input_dim)
    
    # Dummy inputs
    batch_size = 4
    text_ids = torch.randint(0, 1000, (batch_size, 50))
    text_mask = torch.ones((batch_size, 50))
    images = torch.randn(batch_size, 3, 224, 224)
    tabular_data = torch.randn(batch_size, tabular_input_dim)
    
    output = model(text_ids, text_mask, images, tabular_data)
    # print(f"Output shape: {output.shape}")