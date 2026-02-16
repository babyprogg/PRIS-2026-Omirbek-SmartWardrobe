import json

class NLPEngine:
    def __init__(self, rules_path="data/raw/rules.json"):
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f).get("keywords", {})

    def extract_intent(self, user_text):
        """Определяет стиль одежды по тексту пользователя"""
        user_text = user_text.lower()
        
        # Простейший алгоритм: ищем совпадения ключевых слов
        for category, keywords in self.rules.items():
            for word in keywords:
                if word in user_text:
                    return category
        
        return "Casual"  # По умолчанию, если ничего не понятно