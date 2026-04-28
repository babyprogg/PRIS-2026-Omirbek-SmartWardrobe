import json
import os

class Garment:
    def __init__(self, name, category, color):
        self.name = name
        self.category = category
        self.color = color

class StyleExpert:
    def __init__(self, rules_path="data/raw/rules.json"):
        self.rules_path = rules_path
        self.rules = self.load_rules()

    def load_rules(self):
        """Загрузка правил из внешнего JSON файла"""
        if os.path.exists(self.rules_path):
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"color_matches": {}, "occasions": {}}

    @staticmethod
    def _norm_color(color):
        return (color or "").strip().capitalize()

    def check_combination(self, item1, item2):
        # Достаем список сочетаний из загруженного JSON
        matches = self.rules.get("color_matches", {})
        c1 = self._norm_color(item1.color)
        c2 = self._norm_color(item2.color)
        
        if c1 in matches.get(c2, []) or c2 in matches.get(c1, []):
            return True, f"✅ {c1} и {c2} отлично сочетаются по правилам стиля."
        
        if c1 == c2:
            return True, "⚪ Монохромный образ выглядит собранно и современно."
        
        return False, f"⚠️ Пара {c1} + {c2} может выглядеть спорно. Лучше добавить нейтральный слой или аксессуар."

    def get_recommendation(self, occasion):
        # Достаем советы из загруженного JSON
        recommendations = self.rules.get("occasions", {})
        base = recommendations.get(occasion, "Ориентируйтесь на погоду и комфорт.")
        extra = {
            "Свадьба": " Добавьте аккуратные аксессуары и избегайте полностью белого образа.",
            "Работа": " Сделайте акцент на чистых линиях и спокойной палитре.",
            "Прогулка": " Выбирайте дышащие материалы и удобную обувь.",
        }
        return base + extra.get(occasion, "")