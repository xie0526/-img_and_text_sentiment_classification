import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms

def load_labels(train_file):
    labels = []
    guids = []
    with open(train_file, 'r') as f:
        next(f)  # 跳过第一行（标题行）
        for line in f:
            guid, label = line.strip().split(',')
            guids.append(guid)
            labels.append(label)
    return guids, labels

def load_text(guid, txt_dir):
    txt_path = os.path.join(txt_dir, f'{guid}.txt')
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

def load_image(guid, img_dir):
    img_path = os.path.join(img_dir, f'{guid}.jpg')
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return None
    image = Image.open(img_path).convert('RGB')
    return image

def load_all_texts(txt_dir, guids):
    text_dict = {}
    for guid in guids:
        text_dict[guid] = load_text(guid, txt_dir)
    return text_dict

class MultimodalDataset(Dataset):
    def __init__(self, txt_dir, img_dir, guids, labels, tokenizer, transform=None):
        self.txt_dir = txt_dir
        self.img_dir = img_dir
        self.gids = guids
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform
        self.text_data = load_all_texts(txt_dir, guids)

    def __len__(self):
        return len(self.gids)

    def label_to_int(self, label):
        """将情感标签转换为整数"""
        if label == 'positive':
            return 2
        elif label == 'neutral':
            return 1
        elif label == 'negative':
            return 0
        else:
            raise ValueError(f"Unknown label: {label}")

    def __getitem__(self, idx):
        guid = self.gids[idx]
        label = self.labels[idx]
        
        # 获取文本数据
        text = load_text(guid, self.txt_dir)
        
        # 获取图像数据
        image = load_image(guid, self.img_dir)
        
        # 文本处理（分词和编码）
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        
        if self.transform and image is not None:
            image = self.transform(image)
        
        label_int = self.label_to_int(label)
        
        if image is None:
            image = torch.zeros(3, 224, 224)  # 用零填充（可以根据实际需求调整）
        
        return {
            'guid': guid,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image,
            'label': label_int
        }
