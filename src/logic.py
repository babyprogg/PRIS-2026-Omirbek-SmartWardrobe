# src/logic.py

class Garment:
    """Объектная модель предмета одежды (Лаба №3)"""
    def __init__(self, name, category, color):
        self.name = name
        self.category = category  # например, 'Top', 'Bottom', 'Shoes'
        self.color = color

class StyleExpert:
    """Машина вывода на правилах (Лаба №2)"""
    def __init__(self):
        # База знаний: какие цвета сочетаются
        self.color_matches = {
            "Белый": ["Черный", "Синий", "Красный", "Зеленый"],
            "Черный": ["Белый", "Серый", "Красный", "Синий"],
            "Синий": ["Белый", "Серый", "Бежевый"],
            "Бежевый": ["Синий", "Коричневый", "Белый"]
        }

    def check_combination(self, item1, item2):
        """Проверка сочетаемости двух вещей по правилам if-then"""
        if item1.color in self.color_matches.get(item2.color, []) or \
           item2.color in self.color_matches.get(item1.color, []):
            return True, "✅ Эти цвета отлично сочетаются!"
        
        if item1.color == item2.color:
            return True, "⚪ Монохромный образ — это стильно."
        
        return False, "⚠️ Сочетание может выглядеть спорно."

    def get_recommendation(self, occasion):
        """Правила для дресс-кода (Лаба №2)"""
        rules = {
            "Свадьба": "Рекомендуется: Костюм или формальное платье. Избегайте чисто белого.",
            "Работа": "Рекомендуется: Рубашка, брюки или юбка карандаш. Деловой стиль.",
            "Прогулка": "Рекомендуется: Футболка, джинсы и удобная обувь. Casual."
        }
        return rules.get(occasion, "Просто оденьтесь по погоде!")