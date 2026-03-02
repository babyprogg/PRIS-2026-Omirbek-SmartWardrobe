import json
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

class NLPEngine:
    def __init__(self, rules_path="data/raw/rules.json"):
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f).get("keywords", {})
        
        # Инициализация компонентов Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def extract_intent(self, user_text):
        """Определяет стиль одежды по тексту пользователя с использованием лемматизации"""
        doc = Doc(user_text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        
        # Лемматизируем каждое слово
        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemmas.append(token.lemma.lower())
        
        # Сравнение лемм с ключевыми словами
        for category, keywords in self.rules.items():
            for word in keywords:
                if word.lower() in lemmas:
                    return category
        
        return "Casual"  # По умолчанию