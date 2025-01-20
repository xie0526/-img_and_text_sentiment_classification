import torch
import torch.nn as nn
from sklearn.metrics import classification_report

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    criterion = nn.CrossEntropyLoss()  # 损失函数
    
    with torch.no_grad():
        for batch in test_loader:
            # 处理不同的模型类型
            if isinstance(model, TextOnlyModel):  # 处理文本模型
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
            elif isinstance(model, ImageOnlyModel):  # 处理图像模型
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
            elif isinstance(model, MultimodalModel):  # 处理多模态模型
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, images)
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 计算预测结果
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            
            # 保存所有预测值和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct_preds / total_preds
    avg_loss = total_loss / len(test_loader)
    
    print(f"Evaluation Loss: {avg_loss:.4f}, Evaluation Accuracy: {accuracy:.4f}")
    
    # 输出分类报告
    report = classification_report(all_labels, all_preds, zero_division=1, target_names=["negative", "neutral", "positive"])
    print(report)
    
    # 返回评估结果
    return accuracy, avg_loss, report


# 这里是模型加载和评估部分（假设模型已经在某处定义和训练）
def evaluate_models(device, val_loader):
    
    text_only_model = TextOnlyModel(text_encoder=text_encoder).to(device)
    image_only_model = ImageOnlyModel(image_encoder=image_encoder).to(device)
    multimodal_model = MultimodalModel(text_encoder=text_encoder, image_encoder=image_encoder).to(device)

    # 加载模型权重
    text_only_model.load_state_dict(torch.load("best_text_only_model.pth"))
    image_only_model.load_state_dict(torch.load("best_image_only_model.pth"))
    multimodal_model.load_state_dict(torch.load("best_multimodal_model.pth"))

    # 评估文本单模态模型
    print("Evaluating Text Only Model...")
    evaluate(text_only_model, val_loader, device)

    # 评估图像单模态模型
    print("Evaluating Image Only Model...")
    evaluate(image_only_model, val_loader, device)

    # 评估多模态模型
    print("Evaluating Multimodal Model...")
    evaluate(multimodal_model, val_loader, device)

