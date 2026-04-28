import json
import math
import re

import numpy as np

from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Segmenter,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class NLPEngine:
    def __init__(self, rules_path="data/raw/rules.json"):
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        self.rules = payload.get("keywords", {}) or {}

        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

        self.intent_prototypes = self._build_intent_prototypes()
        self.semantic_backend = "tfidf"
        self.semantic_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.semantic_model = None
        self.intent_proto_embeddings = None

        self.word_vectorizer = None
        self.char_vectorizer = None
        self.proto_word_centroids = {}
        self.proto_char_centroids = {}
        self._init_semantic_backend()

        self.color_vocab = {
            "белый": "Белый",
            "черный": "Черный",
            "чёрный": "Черный",
            "синий": "Синий",
            "голубой": "Синий",
            "бежевый": "Бежевый",
            "серый": "Серый",
            "красный": "Красный",
            "зеленый": "Зеленый",
            "зелёный": "Зеленый",
        }

        self.material_vocab = {
            "хлопок": "Хлопок",
            "шерсть": "Шерсть",
            "шёрсть": "Шерсть",
            "деним": "Деним",
            "кожа": "Кожа",
            "полиэстер": "Полиэстер",
            "холст": "Холст",
            "шелк": "Шёлк",
            "шёлк": "Шёлк",
            "лен": "Лен",
            "лён": "Лен",
        }

    def _init_semantic_backend(self):
        if SentenceTransformer is None:
            self._init_tfidf_models()
            return

        try:
            self.semantic_model = SentenceTransformer(self.semantic_model_name)
            self.intent_proto_embeddings = self._build_proto_embeddings()
            if self.intent_proto_embeddings:
                self.semantic_backend = "sbert"
                return
        except Exception:
            self.semantic_model = None
            self.intent_proto_embeddings = None

        self._init_tfidf_models()

    def _build_intent_prototypes(self):
        base = {
            "Formal": [
                "деловая встреча в офисе",
                "строгий стиль для работы",
                "официальное мероприятие",
                "business meeting formal outfit",
            ],
            "Casual": [
                "повседневная прогулка по городу",
                "комфортный городской стиль",
                "встреча с друзьями в кафе",
                "everyday casual look",
            ],
            "Sport": [
                "тренировка в спортзале",
                "одежда для бега",
                "фитнес и активный отдых",
                "sport training outfit",
            ],
            "Smart Casual": [
                "умный повседневный стиль",
                "сдержанно и стильно для ужина",
                "smart casual на встречу",
                "smart casual balanced look",
            ],
        }

        for intent, keywords in self.rules.items():
            if intent not in base:
                base[intent] = []
            base[intent].extend(keywords)
        return base

    def _build_proto_embeddings(self):
        if self.semantic_model is None:
            return None

        embeddings = {}
        for intent, docs in self.intent_prototypes.items():
            docs = [d.strip() for d in docs if d and str(d).strip()]
            if not docs:
                continue
            try:
                doc_emb = self.semantic_model.encode(docs, normalize_embeddings=True)
            except Exception:
                continue
            embeddings[intent] = np.mean(doc_emb, axis=0)

        return embeddings or None

    def _init_tfidf_models(self):
        (
            self.word_vectorizer,
            self.char_vectorizer,
            self.proto_word_centroids,
            self.proto_char_centroids,
        ) = self._fit_prototype_models()
        self.semantic_backend = "tfidf"

    def _fit_prototype_models(self):
        all_docs = []
        intent_ranges = {}
        cursor = 0

        for intent, docs in self.intent_prototypes.items():
            docs = [d.strip() for d in docs if d and d.strip()]
            if not docs:
                continue
            start = cursor
            all_docs.extend(docs)
            cursor += len(docs)
            intent_ranges[intent] = (start, cursor)

        if not all_docs:
            return None, None, {}, {}

        word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)

        word_matrix = word_vectorizer.fit_transform(all_docs)
        char_matrix = char_vectorizer.fit_transform(all_docs)

        proto_word_centroids = {}
        proto_char_centroids = {}

        for intent, (start, end) in intent_ranges.items():
            proto_word_centroids[intent] = word_matrix[start:end].mean(axis=0)
            proto_char_centroids[intent] = char_matrix[start:end].mean(axis=0)

        return word_vectorizer, char_vectorizer, proto_word_centroids, proto_char_centroids

    def _extract_lemmas(self, user_text):
        doc = Doc(user_text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        lemmas = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemma = (token.lemma or "").lower().strip()
            if lemma:
                lemmas.append(lemma)
        return lemmas

    def _keyword_scores(self, lemmas):
        lemma_set = set(lemmas)
        scores = {}
        matches = {}
        for intent, keywords in self.rules.items():
            hit_words = [k for k in keywords if str(k).lower() in lemma_set]
            total = max(1, len(keywords))
            scores[intent] = len(hit_words) / total
            matches[intent] = hit_words
        return scores, matches

    def _semantic_scores(self, text):
        if not text.strip():
            return {}

        if self.semantic_backend == "sbert" and self.intent_proto_embeddings:
            try:
                query_vec = self.semantic_model.encode([text], normalize_embeddings=True)[0]
            except Exception:
                return {}
            scores = {}
            for intent, proto in self.intent_proto_embeddings.items():
                scores[intent] = max(0.0, float(np.dot(query_vec, proto)))
            return scores

        if self.word_vectorizer is None or self.char_vectorizer is None:
            return {}

        query_word = self.word_vectorizer.transform([text])
        query_char = self.char_vectorizer.transform([text])

        scores = {}
        for intent in self.proto_word_centroids.keys():
            w = float(cosine_similarity(query_word, self.proto_word_centroids[intent])[0, 0])
            c = float(cosine_similarity(query_char, self.proto_char_centroids[intent])[0, 0])
            scores[intent] = 0.6 * w + 0.4 * c
        return scores

    @staticmethod
    def _normalize_distribution(score_map):
        if not score_map:
            return {}
        max_v = max(score_map.values())
        exps = {k: math.exp(v - max_v) for k, v in score_map.items()}
        s = sum(exps.values()) or 1.0
        return {k: exps[k] / s for k in exps}

    def extract_intent_with_confidence(self, user_text):
        lemmas = self._extract_lemmas(user_text)
        if not lemmas:
            return {
                "intent": "Casual",
                "confidence": 0.35,
                "matched_keywords": [],
                "lemma_count": 0,
                "intent_distribution": {"Casual": 1.0},
            }

        keyword_scores, matches = self._keyword_scores(lemmas)
        semantic_scores = self._semantic_scores(" ".join(lemmas))

        intents = set(keyword_scores.keys()) | set(semantic_scores.keys())
        if not intents:
            intents = {"Casual"}

        final_scores = {}
        for intent in intents:
            k = keyword_scores.get(intent, 0.0)
            s = semantic_scores.get(intent, 0.0)
            # Гибридный скор: semantic similarity + domain keyword evidence.
            final_scores[intent] = 0.7 * s + 0.3 * k

        distribution = self._normalize_distribution(final_scores)
        best_intent = max(distribution, key=distribution.get)
        best_conf = float(distribution[best_intent])

        return {
            "intent": best_intent,
            "confidence": round(max(0.35, min(0.98, best_conf)), 2),
            "matched_keywords": matches.get(best_intent, []),
            "lemma_count": len(set(lemmas)),
            "intent_distribution": {
                k: round(v, 3)
                for k, v in sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            },
        }

    def _extract_constraints(self, user_text, lemmas):
        text = (user_text or "").lower()
        lemma_set = set(lemmas)

        preferred_colors = []
        avoid_colors = []
        for lemma, normalized in self.color_vocab.items():
            if lemma not in lemma_set:
                continue
            if re.search(rf"(без|избегай|избегать|не\s+хочу)\s+{lemma}", text):
                avoid_colors.append(normalized)
            else:
                preferred_colors.append(normalized)

        preferred_materials = []
        avoid_materials = []
        for lemma, normalized in self.material_vocab.items():
            if lemma not in lemma_set:
                continue
            if re.search(rf"(без|избегай|избегать|не\s+хочу)\s+{lemma}", text):
                avoid_materials.append(normalized)
            else:
                preferred_materials.append(normalized)

        temp = None
        m = re.search(r"(-?\d{1,2})\s*(?:°|c|град)", text)
        if m:
            temp = int(m.group(1))

        return {
            "preferred_colors": sorted(set(preferred_colors)),
            "avoid_colors": sorted(set(avoid_colors)),
            "preferred_materials": sorted(set(preferred_materials)),
            "avoid_materials": sorted(set(avoid_materials)),
            "temperature_hint": temp,
        }

    def analyze_text(self, user_text):
        intent_info = self.extract_intent_with_confidence(user_text)
        lemmas = self._extract_lemmas(user_text)
        constraints = self._extract_constraints(user_text, lemmas)
        return {
            "intent": intent_info["intent"],
            "confidence": intent_info["confidence"],
            "matched_keywords": intent_info.get("matched_keywords", []),
            "intent_distribution": intent_info.get("intent_distribution", {}),
            "constraints": constraints,
        }

    def extract_intent(self, user_text):
        return self.extract_intent_with_confidence(user_text)["intent"]
