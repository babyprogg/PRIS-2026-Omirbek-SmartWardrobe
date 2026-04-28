import cv2
import numpy as np


class CVEngine:
    def __init__(self):
        # OCR intentionally disabled: Vision Lab focuses on visual similarity only.
        self.ocr_enabled = False

    def process_image(self, image_bytes):
        """
        Обработка изображения:
        1. Выделение центрального объекта
        2. Определение доминирующего цвета
        3. Формирование безопасных подсказок (без OCR)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None

        h, w, _ = image.shape
        center_crop = image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        avg_color_bgr = np.mean(center_crop, axis=(0, 1))

        color_name = self._get_color_name(avg_color_bgr)
        color_conf = self._color_confidence(center_crop)

        return {
            "color": color_name,
            "color_confidence": color_conf,
            "suggested_category": "Top",
            "suggested_style": "Casual",
            "avg_bgr": avg_color_bgr.tolist(),
            "ocr_enabled": self.ocr_enabled,
        }

    @staticmethod
    def _color_confidence(image_bgr):
        # Низкая дисперсия на объекте -> выше уверенность в доминирующем цвете.
        std = float(np.std(image_bgr))
        conf = max(0.35, min(0.98, 1.0 - std / 160.0))
        return round(conf, 2)

    def _get_color_name(self, bgr):
        b, g, r = bgr
        if r > 205 and g > 205 and b > 205:
            return "Белый"
        if r < 55 and g < 55 and b < 55:
            return "Черный"
        if abs(r - g) < 20 and abs(g - b) < 20 and r < 190:
            return "Серый"
        if r > 145 and g > 130 and b < 120:
            return "Бежевый"
        if r > g + 25 and r > b + 25:
            return "Красный"
        if b > g + 20 and b > r + 20:
            return "Синий"
        if g > r + 20 and g > b + 20:
            return "Зеленый"
        return "Серый"
