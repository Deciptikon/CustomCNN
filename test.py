from cProfile import label
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from CustomCNN import CustomCNN, EfficientCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR_BORDER = (0, 0, 0)
W_BORDER = 3
PADDING = 30

"""Настройки нейосети"""
class_names = ['airplane', 'bird', 'drone', 'none', 'uav']
model = EfficientCNN(num_classes=5) 
weights_path = "best_model.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))

print("Параметры модели")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel() * param.element_size() / 1024**2:.2f} MB")

# Трансформации
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def dist(point_1, point_2):
    d2 = (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2
    return d2 ** 0.5

def update_trajectory(trajectory, contours):
    if len(contours) == 0:
        #print("len(contours) == 0")
        return trajectory, []

    if len(trajectory) == 0:
        #print("len(trajectory) == 0")
        x, y, w, h = cv2.boundingRect(contours[0])
        trajectory.append((x + w/2, y + h/2))
        return trajectory, []

    min_area = 5
    min_d = 100000
    x, y, w, h = cv2.boundingRect(contours[0])
    p = (x + w/2, y + h/2)
    p_w, p_h = w/2, h/2

    print("Количество контуров: ", len(contours))
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            d = dist((x + w/2, y + h/2), trajectory[-1])
            if d < min_d:
                print("d < min_d : ", d)
                min_d = d
                p = (x + w/2, y + h/2)
                p_w = w/2
                p_h = h/2
    P = []

    if min_d < 800:
        print(p)
        trajectory.append(p)
        P = [p, (p_w, p_h)]
    
    return trajectory, P

def predict_single_image(model, img, transform, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Предсказывает класс для одного изображения
    Args:
        model: обученная модель
        image: изображение
        transform: трансформации для preprocessing
        class_names: список названий классов
        device: устройство для вычислений
    Returns:
        pred_class (str): название предсказанного класса
        confidence (float): уверенность модели (0-1)
    """
    # Загрузка и преобразование изображения
    img = transform(img).unsqueeze(0).to(device)  # Добавляем batch-размер
    
    # Предсказание
    
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        pred_class = class_names[pred_idx.item()]
    
    return pred_class, confidence.item()

def init_background_subtractor(history=500, threshold=16):
    """Инициализация фонового вычитателя"""
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=threshold,
        detectShadows=False
    )

def apply_strong_denoising(fgmask):
    """
    Улучшенное подавление шумов с учетом артефактов сжатия MP4
    """
    # 1. Увеличиваем размер ядра для морфологических операций
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 2. Последовательное применение открытия и закрытия
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Дополнительная медианная фильтрация
    fgmask = cv2.medianBlur(fgmask, 5)
    
    # 4. Пороговая фильтрация для удаления слабых остаточных шумов
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    return fgmask

def process_frame(frame, fgbg):
    """Обработка кадра: вычитание фона + морфология"""
    fgmask = fgbg.apply(frame)

    fgmask = apply_strong_denoising(fgmask)
    
    return fgmask

def draw_contours(frame, contours, min_area=5):
    """Отрисовка контуров движущихся объектов"""
    
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BORDER, W_BORDER)
    return frame

def draw_trajectory(image, points, color=(0, 0, 255), thickness=2):
    """
    Рисует траекторию по заданным точкам на изображении
    
    :param image: исходное изображение (numpy array)
    :param points: список точек в формате [(x1, y1), (x2, y2), ...]
    :param color: цвет траектории в формате BGR (по умолчанию красный)
    :param thickness: толщина линии (по умолчанию 2)
    :return: изображение с нарисованной траекторией
    """
    if len(points) < 2:
        return image  # Нельзя нарисовать линию из одной точки
    
    # Конвертируем точки в numpy array для удобства
    pts = np.array(points, np.int32)
    
    # Рисуем полилинию
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)
    
    # Рисуем кружки в точках траектории
    for point in points:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image, center, radius=3, color=color, thickness=-1)
    
    return image

def main(video_path):
    """Основной цикл обработки видео"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: видео не загружено!")
        return
    
    
    points_trajectory = []
    
    fgbg = init_background_subtractor()

    model.to(device)
    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fgmask = process_frame(frame, fgbg)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points_trajectory, p = update_trajectory(points_trajectory, contours)
        
        if len(p)>0:
            xy, wh = p
            x,y = xy
            w,h = wh
            cap_img = frame[int(y-h-PADDING):int(y+h+PADDING), int(x-w-PADDING):int(x+w+PADDING)]

            # Конвертируем OpenCV -> PIL
            cap_img_rgb = cv2.cvtColor(cap_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            pil_image = Image.fromarray(cap_img_rgb)  # numpy -> PIL

            pred, conf = predict_single_image(model, pil_image, transform, class_names)
            print(pred, conf)
            text = f"{pred} : {int(conf*100.0)}"
            cv2.putText(frame, 
                        text, 
                        (int(x - w), int(y - h - 5)),  # Позиция текста (над прямоугольником)
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7,  # Размер шрифта
                        COLOR_BORDER,  # Цвет (зелёный)
                        W_BORDER)  # Толщина линии
        
        #print(points_trajectory)
        frame_with_contours = draw_trajectory(frame.copy(), points_trajectory)
        frame_with_contours = draw_contours(frame_with_contours, contours) # frame.copy()
        
        # Вывод результата
        #cv2.imshow("Original", frame)
        #cv2.imshow("Foreground Mask", fgmask)
        cv2.imshow("Detected Objects", frame_with_contours)
        
        if cv2.waitKey(30) == 27:  # Выход по ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test_UFO.mp4"

    main(video_path)