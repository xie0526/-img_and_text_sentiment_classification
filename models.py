from transformers import BertModel
import torch
import torch.nn as nn
import torchvision.models as models

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        return x

class MultimodalModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, hidden_size=512, num_classes=3, dropout=0.5):
        super(MultimodalModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fc1 = nn.Linear(2048 + 768, hidden_size)  # 图像特征2048 + 文本特征768
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(image)
        fused_features = torch.cat((text_features, image_features), dim=1)
        x = self.fc1(fused_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        x = self.fc2(x)
        return x
