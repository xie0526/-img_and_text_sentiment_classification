import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ImageOnlyModel(nn.Module):
    def __init__(self, image_encoder, hidden_size=512, num_classes=3):
        super(ImageOnlyModel, self).__init__()
        self.image_encoder = image_encoder
        self.fc1 = nn.Linear(2048, hidden_size)  # ResNet50的输出特征大小是2048
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 分类层
        self.relu = nn.ReLU()

    def forward(self, image):
        image_features = self.image_encoder(image)
        x = self.fc1(image_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_image_only_model(model, train_loader, val_loader, epochs=10, lr=1e-5, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    # 训练过程中使用学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
        
        train_accuracy = correct_preds / total_preds
        train_loss = total_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct_preds = 0
        val_total_preds = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct_preds += (preds == labels).sum().item()
                val_total_preds += labels.size(0)
        
        val_accuracy = val_correct_preds / val_total_preds
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            # 保存当前最佳模型
            torch.save(model.state_dict(), "best_image_only_model.pth")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping!")
                break

        # 更新学习率
        scheduler.step()

    # 最终保存模型
    torch.save(model.state_dict(), "final_image_only_model.pth")

# 训练示例：
# image_encoder = models.resnet50(pretrained=True)  # 加载预训练的ResNet50
# image_only_model = ImageOnlyModel(image_encoder=image_encoder).to(device)
# train_image_only_model(image_only_model, train_loader, val_loader, epochs=10, lr=1e-5, patience=3)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ImageOnlyModel(nn.Module):
    def __init__(self, image_encoder, hidden_size=512, num_classes=3):
        super(ImageOnlyModel, self).__init__()
        self.image_encoder = image_encoder
        self.fc1 = nn.Linear(2048, hidden_size)  # ResNet50的输出特征大小是2048
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 分类层
        self.relu = nn.ReLU()

    def forward(self, image):
        image_features = self.image_encoder(image)
        x = self.fc1(image_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_image_only_model(model, train_loader, val_loader, epochs=10, lr=1e-5, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    # 训练过程中使用学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
        
        train_accuracy = correct_preds / total_preds
        train_loss = total_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct_preds = 0
        val_total_preds = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct_preds += (preds == labels).sum().item()
                val_total_preds += labels.size(0)
        
        val_accuracy = val_correct_preds / val_total_preds
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            # 保存当前最佳模型
            torch.save(model.state_dict(), "best_image_only_model.pth")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping!")
                break

        # 更新学习率
        scheduler.step()

    # 最终保存模型
    torch.save(model.state_dict(), "final_image_only_model.pth")
