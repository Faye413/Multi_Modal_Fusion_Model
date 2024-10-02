import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from transformers import BertTokenizer

class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, text_column, img_column, tabular_columns, label_column, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.text_column = text_column
        self.img_column = img_column
        self.tabular_columns = tabular_columns
        self.label_column = label_column
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Process text
        text = row[self.text_column]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        text_ids = encoding['input_ids'].squeeze(0)
        text_mask = encoding['attention_mask'].squeeze(0)
        
        # Process image
        img_path = f"{self.img_dir}/{row[self.img_column]}"
        image = Image.open(img_path).convert('RGB')
        image = self.transform_image(image)
        
        # Process tabular data
        tabular_data = torch.tensor(row[self.tabular_columns].values.astype(np.float32))
        
        # Get label
        label = torch.tensor(row[self.label_column])
        
        return {
            'text_ids': text_ids,
            'text_mask': text_mask,
            'image': image,
            'tabular_data': tabular_data,
            'label': label
        }
    
    def transform_image(self, image):
        # Define your image transformations here
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

# Usage example
if __name__ == "__main__":
    csv_file = 'path/to/your/data.csv'
    img_dir = 'path/to/your/images'
    text_column = 'text_description'
    img_column = 'image_filename'
    tabular_columns = ['feature1', 'feature2', 'feature3']
    label_column = 'target'
    
    dataset = MultiModalDataset(csv_file, img_dir, text_column, img_column, tabular_columns, label_column)
    
    sample = dataset[0]
    # print(f"Text IDs shape: {sample['text_ids'].shape}")
    # print(f"Text mask shape: {sample['text_mask'].shape}")
    # print(f"Image shape: {sample['image'].shape}")
    # print(f"Tabular data shape: {sample['tabular_data'].shape}")
    # print(f"Label: {sample['label']}")