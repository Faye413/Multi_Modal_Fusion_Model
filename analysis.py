import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz

class FeatureImportanceAnalyzer:
    def __init__(self, model):
        self.model = model
        self.ig = IntegratedGradients(self.model)
        self.lig_text = LayerIntegratedGradients(self.model, self.model.text_encoder.bert.embeddings)
    
    def analyze_tabular_features(self, inputs, target=None, n_steps=50):
        text_ids, text_mask, images, tabular_data = inputs
        attr = self.ig.attribute((text_ids, text_mask, images, tabular_data), target=target, n_steps=n_steps)
        return attr[3]  # Return attributions for tabular features
    
    def analyze_text_features(self, inputs, target=None, n_steps=50):
        text_ids, text_mask, images, tabular_data = inputs
        attr = self.lig_text.attribute((text_ids, text_mask), target=target, n_steps=n_steps)
        return attr
    
    def visualize_tabular_importance(self, attributions, feature_names):
        attr_sum = attributions.sum(0)
        attr_norm = attr_sum / np.linalg.norm(attr_sum)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_names)), attr_norm)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Tabular Feature Importance')
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def visualize_text_importance(self, attributions, text_ids, tokenizer):
        attr_sum = attributions.sum(dim=2).squeeze(0)
        attr = attr_sum / torch.norm(attr_sum)
        
        tokens = tokenizer.convert_ids_to_tokens(text_ids.squeeze().tolist())
        
        viz.visualize_text_attr(attr, tokens)

# Usage example
if __name__ == "__main__":
    from multi_modal_fusion_model import MultiModalFusionModel
    from transformers import BertTokenizer
    
    num_classes = 10
    tabular_input_dim = 20
    model = MultiModalFusionModel(num_classes, tabular_input_dim)
    analyzer = FeatureImportanceAnalyzer(model)
    
    # Dummy inputs
    batch_size = 1
    text_ids = torch.randint(0, 1000, (batch_size, 50))
    text_mask = torch.ones((batch_size, 50))
    images = torch.randn(batch_size, 3, 224, 224)
    tabular_data = torch.randn(batch_size, tabular_input_dim)
    
    # Analyze tabular features
    tabular_attr = analyzer.analyze_tabular_features((text_ids, text_mask, images, tabular_data))
    feature_names = [f'Feature {i}' for i in range(tabular_input_dim)]
    analyzer.visualize_tabular_importance(tabular_attr.detach().numpy(), feature_names)
    
    # Analyze text features
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_attr = analyzer.analyze_text_features((text_ids, text_mask, images, tabular_data))
    analyzer.visualize_text_importance(text_attr, text_ids, tokenizer)