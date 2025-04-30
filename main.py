import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import random

from CustomCNN import CustomCNN

# 1. Подготовка модели
model = CustomCNN(num_classes=10) 

# 2. Подготовка данных (пример для CIFAR-10)
def random_pixelate(img):
    """Случайная пикселизация: уменьшение и обратное увеличение"""
    # Случайный выбор размера уменьшения (от 8x8 до 64x64)
    downscale = random.choice([8, 16, 24, 32, 48, 64])
    return img.resize((downscale, downscale), Image.NEAREST).resize((128, 128), Image.NEAREST)

transform = transforms.Compose([
    # 1. Стандартные аугментации (на полном размере)
    transforms.RandomRotation(degrees=15),
    #transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),  # Случайные аффинные преобразования
    #transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Перспективные искажения
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Случайное "стирание"
    
    
    # 2. Случайная пикселизация
    transforms.Lambda(random_pixelate),
    
    # 3. Гарантированный ресайз и конвертация
    transforms.Resize((128, 128)),  # На случай если входные размеры отличаются
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Загрузка данных из ваших папок
train_data = datasets.ImageFolder(
    root='./data/train',  # Путь к папке train
    transform=transform
)

test_data = datasets.ImageFolder(
    root='./data/test',   # Путь к папке test
    transform=transform
)

# Создание DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False)

# 3. Настройка обучения
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Функция обучения
def train(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    best_val_acc = 0.0
    best_train_acc = 0.0
    train_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Обучение
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Статистика обучения
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        
        # Валидация
        val_acc = evaluate(model, test_loader, criterion)
        val_accuracies.append(val_acc)
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc

            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}')
        print(f'Val Acc: {val_acc:.2f}% | Train Acc: {train_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}% | Best Val Acc: {best_train_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, val_accuracies

def evaluate(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return accuracy

# 6. Запуск обучения
train_losses, val_accuracies = train(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    epochs=100
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()