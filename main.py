import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

from CustomCNN import CustomCNN, EfficientCNN

# 1. Подготовка модели
model = EfficientCNN(num_classes=5) 

# 2. Подготовка данных
def random_pixelate(img):
    """Случайная пикселизация: уменьшение и обратное увеличение"""
    # Случайный выбор размера уменьшения (от 8x8 до 64x64)
    downscale = random.choice([8, 16, 24, 32, 48, 64])
    return img.resize((downscale, downscale), Image.NEAREST).resize((128, 128), Image.NEAREST)

transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    #transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),  # Случайные аффинные преобразования
    #transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Перспективные искажения
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Случайное "стирание"
    
    
    # Случайная пикселизация
    transforms.Lambda(random_pixelate),
    
    # Гарантированный ресайз и конвертация
    transforms.Resize((128, 128)),  # На случай если входные размеры отличаются
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Загрузка данных из папок
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
        print(f'Curr Val Acc: {val_acc:.2f}% | Curr Train Acc: {train_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}% | Best Train Acc: {best_train_acc:.2f}%')
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

def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=200)
    plt.close()

def plot_image_confusion_matrix(model, test_loader, class_names, n_samples=3):
    model.eval()
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_classes, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Инициализация словаря для примеров
    examples = {i: {j: [] for j in range(n_classes)} for i in range(n_classes)}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            # Конвертируем тензоры в числа
            true_labels = labels.cpu().numpy()
            pred_labels = preds.cpu().numpy()
            
            for img, true, pred in zip(images, true_labels, pred_labels):
                if len(examples[true][pred]) < n_samples:
                    examples[true][pred].append(img.cpu())
    
    # Отрисовка
    for i in range(n_classes):
        for j in range(n_classes):
            ax = axes[i, j]
            ax.axis('off')
            
            # Цвет фона
            bg_color = '#ddffdd' if i == j else '#ffdddd'
            ax.set_facecolor(bg_color)
            
            # Добавление изображений
            if examples[i][j]:
                grid = make_grid(examples[i][j], nrow=1, pad_value=1)
                img_to_show = grid.permute(1, 2, 0).numpy()
                img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())  # Нормализация к [0,1]
                ax.imshow(img_to_show)
            
            # Подписи осей
            if i == n_classes-1:
                ax.set_xlabel(f'Pred: {class_names[j]}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'True: {class_names[i]}', fontsize=9)
    
    plt.savefig('visual_confusion_matrix.png', bbox_inches='tight', dpi=150)
    plt.close()

# 6. Запуск обучения
if True:
    train_losses, val_accuracies = train(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        epochs=100
    )

print("Классы в данных:", len(test_data.classes), test_data.classes)
print("Выходов у модели:", model.fc.out_features)  # Для CNN
#print(model) 

print("Параметры модели")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel() * param.element_size() / 1024**2:.2f} MB")

plot_confusion_matrix(model, test_loader, test_data.classes)
plot_image_confusion_matrix(model, test_loader, test_data.classes)

# 7. Метрики обучения
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

# Сохраняем в файл
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight') 

plt.show()