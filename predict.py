import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms

# 预测数据的 Dataset 类
class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, txt_dir, img_dir, guids, tokenizer, transform=None):
        self.txt_dir = txt_dir
        self.img_dir = img_dir
        self.guids = guids
        self.tokenizer = tokenizer
        self.transform = transform

    def load_text(self, guid):
        txt_path = os.path.join(self.txt_dir, f'{guid}.txt')
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Text file not found: {txt_path}")
            text = ""
        except Exception as e:
            print(f"Error reading {txt_path}: {str(e)}")
            text = ""
        return text

    def load_image(self, guid):
        img_path = os.path.join(self.img_dir, f'{guid}.jpg')
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return None
        image = Image.open(img_path).convert('RGB')
        return image

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, idx):
        guid = self.guids[idx]
        
        # 加载文本和图像
        text = self.load_text(guid)
        image = self.load_image(guid)
        
        # 文本处理
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        
        if self.transform and image is not None:
            image = self.transform(image)
        
        if image is None:
            image = torch.zeros(3, 224, 224)  # 用零填充（可以根据实际需求调整）
        
        return {
            'guid': guid,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image
        }

# 预测函数
def predict(model, test_file, txt_dir, img_dir, tokenizer, batch_size=32):
    # 加载测试集的 GUID
    guids = []
    with open(test_file, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            guid = line.strip().split(',')[0]
            guids.append(guid)

    # 数据预处理：图像和文本
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = PredictDataset(txt_dir=txt_dir, img_dir=img_dir, guids=guids, tokenizer=tokenizer, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 预测
    model.eval()
    predictions = []
    guids_batch = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            guids_batch.extend(batch['guid'])
            
            outputs = model(input_ids, attention_mask, images)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    return guids_batch, predictions

# 保存预测结果
def save_predictions(guids, predictions, output_file="predictions.csv"):
    df = pd.DataFrame({"guid": guids, "label": predictions})
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

