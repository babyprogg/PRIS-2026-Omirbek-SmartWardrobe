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

    def check_combination(self, item1, item2):
        # Достаем список сочетаний из загруженного JSON
        matches = self.rules.get("color_matches", {})
        
        if item1.color in matches.get(item2.color, []) or \
           item2.color in matches.get(item1.color, []):
            return True, "✅ Эти цвета отлично сочетаются!"
        
        if item1.color == item2.color:
            return True, "⚪ Монохромный образ — это стильно."
        
        return False, "⚠️ Сочетание может выглядеть спорно."

    def get_recommendation(self, occasion):
        # Достаем советы из загруженного JSON
        recommendations = self.rules.get("occasions", {})
        return recommendations.get(occasion, "Просто оденьтесь по погоде!")