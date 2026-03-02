import cv2
import numpy as np
import easyocr
import os

class CVEngine:
    def __init__(self):
        # Инициализация EasyOCR (русский и английский)
        # При первом запуске скачает веса (~80МБ)
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)

    def process_image(self, image_bytes):
        """
        Обработка изображения: 
        1. Извлечение текста (EasyOCR)
        2. Определение доминирующего цвета (OpenCV)
        """
        # Декодируем изображение из байтов
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None

        # 1. OCR (поиск текста на бирках)
        results = self.reader.readtext(image)
        detected_text = [res[1] for res in results]

        # 2. Определение цвета (упрощенно)
        # Убираем фон (предполагаем, что объект в центре)
        h, w, _ = image.shape
        center_crop = image[h//4:3*h//4, w//4:3*w//4]
        
        # Средний цвет в формате BGR
        avg_color_bgr = np.mean(center_crop, axis=(0, 1))
        
        # Превращаем в понятное название (очень грубое приближение)
        color_name = self._get_color_name(avg_color_bgr)

        return {
            "text": detected_text,
            "color": color_name,
            "avg_bgr": avg_color_bgr.tolist()
        }

    def _get_color_name(self, bgr):
        b, g, r = bgr
        # Простейшая логика классификации цветов
        if r > 200 and g > 200 and b > 200: return "Белый"
        if r < 50 and g < 50 and b < 50: return "Черный"
        if r > b and r > g: return "Красный"
        if b > r and b > g: return "Синий"
        if g > r and g > b: return "Зеленый"
        if r > 150 and g > 150 and b < 100: return "Бежевый"
        return "Серый"
