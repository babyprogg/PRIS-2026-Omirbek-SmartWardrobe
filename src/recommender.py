import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz, process


class RecommendationEngine:
    """
    Неделя 8: Гибридный рекомендательный движок.
    Использует TF-IDF + Cosine Similarity для поиска похожих вещей
    и FuzzyWuzzy (thefuzz) для нечёткого текстового поиска.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.df = None

    @staticmethod
    def _material_comfort_ranges():
        return {
            "шерсть": (0, 12),
            "wool": (0, 12),
            "кашемир": (-5, 10),
            "cashmere": (-5, 10),
            "деним": (4, 24),
            "denim": (4, 24),
            "хлопок": (12, 32),
            "cotton": (12, 32),
            "полиэстер": (8, 26),
            "polyester": (8, 26),
            "лен": (18, 36),
            "linen": (18, 36),
            "кожа": (5, 22),
            "leather": (5, 22),
            "холст": (14, 32),
            "canvas": (14, 32),
            "шёлк": (14, 28),
            "silk": (14, 28),
        }

    @staticmethod
    def _normalize_category(category: str) -> str:
        c = (category or "").strip().lower()
        fallback = str(category or "").strip()
        mapping = {
            "top": "Top",
            "bottom": "Bottom",
            "shoes": "Shoes",
            "accessory": "Accessory",
            "accessories": "Accessory",
        }
        return mapping.get(c, fallback.title() if fallback else "Top")

    @staticmethod
    def _normalize_style(style: str) -> str:
        s = (style or "").strip().lower()
        mapping = {
            "smart casual": "smart casual",
            "smartcasual": "smart casual",
            "formal": "formal",
            "casual": "casual",
            "sport": "sport",
        }
        return mapping.get(s, s)

    @staticmethod
    def _outerwear_keywords() -> tuple:
        return (
            "тренч",
            "пальто",
            "куртк",
            "плащ",
            "ветровк",
            "парка",
            "пуховик",
            "пиджак",
            "блейзер",
            "джинсовк",
            "кардиган",
            "coat",
            "jacket",
            "trench",
            "parka",
            "blazer",
        )

    @classmethod
    def _is_outerwear_name(cls, name: str) -> bool:
        n = (name or "").strip().lower()
        if not n:
            return False
        return any(key in n for key in cls._outerwear_keywords())

    def _split_tops(self, tops: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if tops.empty:
            return tops, tops
        mask = tops["name"].astype(str).map(self._is_outerwear_name)
        outerwear = tops[mask]
        inner = tops[~mask]
        return inner, outerwear

    @staticmethod
    def _material_temperature_fit(material: str, weather_c: int) -> float:
        material = (material or "").strip().lower()
        comfort = RecommendationEngine._material_comfort_ranges()

        if material not in comfort:
            return 72.0

        min_t, max_t = comfort[material]
        if min_t <= weather_c <= max_t:
            return 100.0

        distance = min(abs(weather_c - min_t), abs(weather_c - max_t))
        return max(45.0, 100.0 - distance * 7.0)

    @staticmethod
    def explain_weather(material: str, weather_c: int) -> str:
        material = (material or "").strip().lower()
        comfort = RecommendationEngine._material_comfort_ranges()
        if material not in comfort:
            return "материал неизвестен системе"

        min_t, max_t = comfort[material]
        if min_t <= weather_c <= max_t:
            return f"комфортно при {weather_c}C (диапазон {min_t}..{max_t}C)"
        if weather_c < min_t:
            return f"может быть прохладно при {weather_c}C (рекомендовано от {min_t}C)"
        return f"может быть жарко при {weather_c}C (рекомендовано до {max_t}C)"

    @staticmethod
    def _style_fit(item_style: str, target_style: str) -> float:
        item_s = RecommendationEngine._normalize_style(item_style)
        target_s = RecommendationEngine._normalize_style(target_style)
        if item_s == target_s:
            return 100.0
        compatible = {
            ("smart casual", "casual"),
            ("casual", "smart casual"),
            ("formal", "smart casual"),
            ("smart casual", "formal"),
        }
        if (item_s, target_s) in compatible:
            return 80.0
        return 55.0

    @staticmethod
    def _color_fit(top_color: str, bottom_color: str, color_rules: dict) -> float:
        if top_color == bottom_color:
            return 90.0
        matches = color_rules or {}
        if top_color in matches.get(bottom_color, []) or bottom_color in matches.get(top_color, []):
            return 100.0
        return 50.0

    @staticmethod
    def _confidence_from_total(total_score: float) -> float:
        # Линейно приводим диапазон к 0..1 с мягкой отсечкой.
        return max(0.0, min(1.0, (total_score - 50.0) / 50.0))

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.8:
            return "High"
        if confidence >= 0.6:
            return "Medium"
        return "Low"

    @staticmethod
    def _build_explanation(style_score: float, color_score: float, weather_score: float) -> str:
        style_part = "стиль хорошо совпадает" if style_score >= 85 else "стиль частично совпадает"
        color_part = "палитра гармонична" if color_score >= 85 else "цветовое сочетание среднее"
        weather_part = "материалы подходят под погоду" if weather_score >= 85 else "по погоде есть компромисс"
        return f"{style_part}; {color_part}; {weather_part}."

    def _apply_context_constraints(self, outfits: pd.DataFrame, constraints: dict | None) -> pd.DataFrame:
        if outfits.empty or not constraints or self.df is None or self.df.empty:
            return outfits

        lookup = self.df.set_index("name", drop=False)
        preferred_colors = set(constraints.get("preferred_colors", []) or [])
        avoid_colors = set(constraints.get("avoid_colors", []) or [])
        preferred_materials = set(constraints.get("preferred_materials", []) or [])
        avoid_materials = set(constraints.get("avoid_materials", []) or [])

        def bonus_for_row(row):
            item_names = [
                row.get("top", ""),
                row.get("bottom", ""),
                row.get("outerwear", ""),
                row.get("shoes", ""),
                row.get("accessory", ""),
            ]
            colors = []
            materials = []

            for name in item_names:
                n = str(name or "").strip()
                if not n or n == "-" or n not in lookup.index:
                    continue
                r = lookup.loc[n]
                colors.append(str(r.get("color", "")))
                materials.append(str(r.get("material", "")))

            bonus = 0.0
            bonus += 5.0 * len(preferred_colors.intersection(colors))
            bonus -= 12.0 * len(avoid_colors.intersection(colors))
            bonus += 4.0 * len(preferred_materials.intersection(materials))
            bonus -= 10.0 * len(avoid_materials.intersection(materials))
            return bonus

        adjusted = outfits.copy()
        adjusted["context_bonus"] = adjusted.apply(bonus_for_row, axis=1).round(1)

        if constraints.get("temperature_hint") is not None:
            hinted = int(constraints["temperature_hint"])
            # Дополнительный бонус, если погодный fit высокий при явной температурной подсказке.
            adjusted["context_bonus"] += ((adjusted["weather_score"] - 75.0) / 8.0).round(1)
            adjusted["temperature_used"] = hinted

        adjusted["base_total_score"] = adjusted["total_score"]
        adjusted["total_score"] = (adjusted["base_total_score"] + adjusted["context_bonus"]).clip(0, 100).round(1)

        adjusted["confidence"] = adjusted["total_score"].apply(self._confidence_from_total).round(2)
        adjusted["uncertainty"] = (1.0 - adjusted["confidence"]).round(2)
        adjusted["confidence_label"] = adjusted["confidence"].apply(self._confidence_label)
        return adjusted.sort_values("total_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _safe_name(value):
        return str(value or "").strip()

    def build_preference_profile(self, feedback_df: pd.DataFrame) -> dict:
        """
        Строит легковесный профиль предпочтений по лайкам/дизлайкам пользователя.
        Это дополняет JSON-правила динамическим компонентом.
        """
        profile = {
            "style": {},
            "color": {},
        }
        if feedback_df is None or feedback_df.empty or self.df is None or self.df.empty:
            return profile

        lookup = self.df.set_index("name", drop=False)

        def add_score(keyspace: str, key: str, delta: int):
            key = self._safe_name(key)
            if not key:
                return
            profile[keyspace][key] = profile[keyspace].get(key, 0) + delta

        for _, event in feedback_df.iterrows():
            delta = 1 if str(event.get("feedback", "")).strip().lower() == "like" else -1
            names = [
                event.get("top", ""),
                event.get("bottom", ""),
                event.get("outerwear", ""),
                event.get("shoes", ""),
                event.get("accessory", ""),
            ]
            for name in names:
                n = self._safe_name(name)
                if not n or n == "-" or n not in lookup.index:
                    continue
                row = lookup.loc[n]
                add_score("style", row.get("style", ""), delta)
                add_score("color", row.get("color", ""), delta)

        return profile

    @staticmethod
    def _preference_fit(item_style: str, item_color: str, profile: dict) -> float:
        if not profile:
            return 75.0

        style_scores = profile.get("style", {})
        color_scores = profile.get("color", {})
        s = style_scores.get(str(item_style), 0)
        c = color_scores.get(str(item_color), 0)
        score = 75.0 + (s * 4.0) + (c * 3.0)
        return max(40.0, min(100.0, score))

    def _item_base_score(
        self,
        row: pd.Series,
        target_style: str,
        weather_c: int,
        preference_profile: dict | None,
    ) -> float:
        style_score = self._style_fit(row.get("style", ""), target_style)
        weather_score = self._material_temperature_fit(row.get("material", ""), weather_c)
        preference_score = self._preference_fit(
            row.get("style", ""),
            row.get("color", ""),
            preference_profile or {},
        )
        return 0.45 * style_score + 0.35 * weather_score + 0.2 * preference_score

    def _rerank_by_base(
        self,
        frame: pd.DataFrame,
        target_style: str,
        weather_c: int,
        preference_profile: dict | None,
        top_k_each: int,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        scored = frame.copy()
        scored["_base"] = scored.apply(
            lambda r: self._item_base_score(r, target_style, weather_c, preference_profile),
            axis=1,
        )
        return scored.sort_values("_base", ascending=False).head(top_k_each).drop(columns=["_base"])

    def _rank_candidates(self, frame: pd.DataFrame, query: str, top_k: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        docs = (
            frame.get("category", "").fillna("") + " "
            + frame.get("color", "").fillna("") + " "
            + frame.get("style", "").fillna("") + " "
            + frame.get("name", "").fillna("")
        )

        if docs.astype(str).str.strip().eq("").all():
            return frame.head(top_k)

        local_vectorizer = TfidfVectorizer()
        try:
            matrix = local_vectorizer.fit_transform(docs)
            query_vec = local_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, matrix).flatten()
        except ValueError:
            return frame.head(top_k)

        ranked = frame.copy()
        ranked["_rank"] = scores
        return ranked.sort_values("_rank", ascending=False).head(top_k).drop(columns=["_rank"])

    def build_index(self, df: pd.DataFrame):
        """
        Строит TF-IDF индекс из DataFrame гардероба.
        Каждая вещь представляется строкой: "<category> <color> <style>".
        """
        self.df = df.copy().reset_index(drop=True)
        if self.df.empty:
            self.tfidf_matrix = None
            return

        for col in ["name", "category", "color", "style", "material"]:
            if col not in self.df.columns:
                self.df[col] = ""

        if "image_path" not in self.df.columns:
            self.df["image_path"] = ""

        self.df["category"] = self.df["category"].map(self._normalize_category)

        # Объединяем текстовые поля в один документ
        self.df["_doc"] = (
            self.df.get("category", "").fillna("") + " " +
            self.df.get("color", "").fillna("") + " " +
            self.df.get("style", "").fillna("") + " " +
            self.df.get("material", "").fillna("") + " " +
            self.df.get("name", "").fillna("")
        )

        if self.df["_doc"].astype(str).str.strip().eq("").all():
            self.tfidf_matrix = None
            return

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_doc"])
        except ValueError:
            self.tfidf_matrix = None

    def find_similar(self, query: str, top_n: int = 3) -> pd.DataFrame:
        """
        Находит top_n вещей, наиболее похожих на текстовый запрос.
        Возвращает DataFrame с колонкой 'similarity'.
        """
        if self.tfidf_matrix is None or self.df is None:
            return pd.DataFrame()

        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        except ValueError:
            return pd.DataFrame()

        top_indices = scores.argsort()[::-1][:top_n]

        result = self.df.iloc[top_indices].copy()
        result["similarity"] = (scores[top_indices] * 100).round(1)
        return result[["name", "category", "color", "material", "style", "image_path", "similarity"]].reset_index(drop=True)

    def find_similar_structured(
        self,
        category: str,
        color: str,
        style: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Поиск с приоритетом категории над цветом и стилем.
        Нужен для Vision Lab: футболка должна быть ближе к футболке, чем к шортам того же цвета.
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        category = self._normalize_category(category)
        color = self._safe_name(color)
        style = self._safe_name(style)

        ranked = self.df.copy()
        ranked["category_score"] = ranked["category"].apply(lambda x: 1.0 if str(x) == category else 0.0)
        ranked["color_score"] = ranked["color"].apply(lambda x: 1.0 if str(x) == color else 0.0)
        ranked["style_score"] = ranked["style"].apply(lambda x: 1.0 if str(x) == style else 0.0)

        ranked["similarity"] = (
            100.0
            * (0.65 * ranked["category_score"] + 0.2 * ranked["style_score"] + 0.15 * ranked["color_score"])
        ).round(1)

        ranked = ranked.sort_values(["similarity", "name"], ascending=[False, True]).head(top_n)
        cols = ["name", "category", "color", "material", "style", "image_path", "similarity"]
        return ranked[cols].reset_index(drop=True)

    def find_near_duplicates(self, threshold: float = 82.0) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        docs = self.df.get("name", "").fillna("") + " " + self.df.get("category", "").fillna("") + " " + self.df.get("color", "").fillna("")
        if docs.astype(str).str.strip().eq("").all():
            return pd.DataFrame()

        vec = TfidfVectorizer()
        try:
            matrix = vec.fit_transform(docs)
        except ValueError:
            return pd.DataFrame()

        sim = cosine_similarity(matrix)

        pairs = []
        n = len(self.df)
        for i in range(n):
            for j in range(i + 1, n):
                score = sim[i, j] * 100.0
                if score >= threshold:
                    pairs.append(
                        {
                            "item_a": self.df.iloc[i]["name"],
                            "item_b": self.df.iloc[j]["name"],
                            "category_a": self.df.iloc[i]["category"],
                            "category_b": self.df.iloc[j]["category"],
                            "similarity": round(float(score), 1),
                        }
                    )

        if not pairs:
            return pd.DataFrame()
        return pd.DataFrame(pairs).sort_values("similarity", ascending=False).reset_index(drop=True)

    def fuzzy_search(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """
        Нечёткий поиск по названию вещи.
        Например, "белая рубашк" найдёт "Белая рубашка" несмотря на опечатку.
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        names = self.df["name"].tolist()
        matches = process.extract(query, names, scorer=fuzz.token_sort_ratio, limit=top_n)

        results = []
        for match_name, score in matches:
            if score >= 40:  # порог схожести
                row = self.df[self.df["name"] == match_name].iloc[0].to_dict()
                row["match_score"] = score
                results.append(row)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        cols = ["name", "category", "color", "material", "style", "match_score"]
        return result_df[[c for c in cols if c in result_df.columns]].reset_index(drop=True)

    def wardrobe_analytics(self) -> dict:
        if self.df is None or self.df.empty:
            return {
                "items_total": 0,
                "style_counts": {},
                "category_counts": {},
                "color_counts": {},
            }

        return {
            "items_total": int(len(self.df)),
            "style_counts": self.df["style"].value_counts().to_dict(),
            "category_counts": self.df["category"].value_counts().to_dict(),
            "color_counts": self.df["color"].value_counts().to_dict(),
        }

    def generate_outfits(
        self,
        target_style: str,
        weather_c: int,
        color_rules: dict,
        top_k_each: int = 6,
        include_outerwear: bool = True,
        include_shoes: bool = True,
        include_accessory: bool = True,
        preference_profile: dict | None = None,
        context_constraints: dict | None = None,
    ) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        df = self.df.copy()
        df["_style_norm"] = df["style"].astype(str).map(self._normalize_style)
        target_norm = self._normalize_style(target_style)

        style_pool = df[df["_style_norm"] == target_norm]
        if style_pool.empty and target_norm in {"formal", "casual"}:
            style_pool = df[df["_style_norm"].isin([target_norm, "smart casual"])]
        if style_pool.empty:
            style_pool = df

        candidate_k = max(top_k_each * 3, top_k_each)

        tops_pool = style_pool[style_pool["category"] == "Top"]
        if tops_pool.empty:
            tops_pool = df[df["category"] == "Top"]
        inner_tops, outerwear = self._split_tops(tops_pool)
        if inner_tops.empty:
            inner_tops = tops_pool
            outerwear = outerwear.iloc[0:0]

        bottoms_pool = style_pool[style_pool["category"] == "Bottom"]
        if bottoms_pool.empty:
            bottoms_pool = df[df["category"] == "Bottom"]

        shoes_pool = style_pool[style_pool["category"] == "Shoes"]
        if shoes_pool.empty:
            shoes_pool = df[df["category"] == "Shoes"]

        accessories_pool = style_pool[style_pool["category"] == "Accessory"]
        if accessories_pool.empty:
            accessories_pool = df[df["category"] == "Accessory"]

        tops = self._rank_candidates(inner_tops, target_style, candidate_k)
        bottoms = self._rank_candidates(bottoms_pool, target_style, candidate_k)
        shoes = self._rank_candidates(shoes_pool, target_style, candidate_k)
        accessories = self._rank_candidates(accessories_pool, target_style, candidate_k)
        outerwear_candidates = self._rank_candidates(outerwear, target_style, candidate_k) if not outerwear.empty else outerwear

        tops = self._rerank_by_base(tops, target_style, weather_c, preference_profile, top_k_each)
        bottoms = self._rerank_by_base(bottoms, target_style, weather_c, preference_profile, top_k_each)
        shoes = self._rerank_by_base(shoes, target_style, weather_c, preference_profile, top_k_each)
        accessories = self._rerank_by_base(accessories, target_style, weather_c, preference_profile, top_k_each)
        outerwear_candidates = self._rerank_by_base(
            outerwear_candidates,
            target_style,
            weather_c,
            preference_profile,
            top_k_each,
        )

        if tops.empty or bottoms.empty:
            return pd.DataFrame()

        combinations = []
        outerwear_rows = [None] if (not include_outerwear or outerwear_candidates.empty) else [row for _, row in outerwear_candidates.iterrows()]
        shoe_rows = [None] if (not include_shoes or shoes.empty) else [row for _, row in shoes.iterrows()]
        accessory_rows = [None] if (not include_accessory or accessories.empty) else [row for _, row in accessories.iterrows()]

        for _, top in tops.iterrows():
            for _, bottom in bottoms.iterrows():
                for outerwear_item in outerwear_rows:
                    for shoe in shoe_rows:
                        for accessory in accessory_rows:
                            style_terms = [
                                self._style_fit(top["style"], target_style),
                                self._style_fit(bottom["style"], target_style),
                                self._style_fit(outerwear_item["style"], target_style) if outerwear_item is not None else 85.0,
                                self._style_fit(shoe["style"], target_style) if shoe is not None else 85.0,
                            ]
                            if accessory is not None:
                                style_terms.append(self._style_fit(accessory["style"], target_style))
                            style_score = sum(style_terms) / len(style_terms)

                            color_score = self._color_fit(top["color"], bottom["color"], color_rules)

                            weather_terms = [
                                self._material_temperature_fit(top.get("material", ""), weather_c),
                                self._material_temperature_fit(bottom.get("material", ""), weather_c),
                                self._material_temperature_fit(outerwear_item.get("material", ""), weather_c) if outerwear_item is not None else 80.0,
                                self._material_temperature_fit(shoe.get("material", ""), weather_c) if shoe is not None else 80.0,
                            ]
                            if accessory is not None:
                                weather_terms.append(self._material_temperature_fit(accessory.get("material", ""), weather_c))
                            weather_score = sum(weather_terms) / len(weather_terms)

                            preference_terms = [
                                self._preference_fit(top.get("style", ""), top.get("color", ""), preference_profile or {}),
                                self._preference_fit(bottom.get("style", ""), bottom.get("color", ""), preference_profile or {}),
                                self._preference_fit(outerwear_item.get("style", ""), outerwear_item.get("color", ""), preference_profile or {}) if outerwear_item is not None else 75.0,
                                self._preference_fit(shoe.get("style", ""), shoe.get("color", ""), preference_profile or {}) if shoe is not None else 75.0,
                            ]
                            if accessory is not None:
                                preference_terms.append(self._preference_fit(accessory.get("style", ""), accessory.get("color", ""), preference_profile or {}))
                            preference_score = sum(preference_terms) / len(preference_terms)

                            versatility = 92.0 if len({top["color"], bottom["color"]}) == 1 else 85.0

                            total = (
                                0.35 * style_score
                                + 0.2 * color_score
                                + 0.25 * weather_score
                                + 0.1 * preference_score
                                + 0.1 * versatility
                            )

                            confidence = self._confidence_from_total(total)
                            uncertainty = 1.0 - confidence
                            explanation = self._build_explanation(style_score, color_score, weather_score)

                            combinations.append(
                                {
                                    "top": top["name"],
                                    "bottom": bottom["name"],
                                    "outerwear": outerwear_item["name"] if outerwear_item is not None else "-",
                                    "shoes": shoe["name"] if shoe is not None else "-",
                                    "accessory": accessory["name"] if accessory is not None else "-",
                                    "style_score": round(style_score, 1),
                                    "color_score": round(color_score, 1),
                                    "weather_score": round(weather_score, 1),
                                    "preference_score": round(preference_score, 1),
                                    "versatility_score": round(versatility, 1),
                                    "total_score": round(total, 1),
                                    "confidence": round(confidence, 2),
                                    "uncertainty": round(uncertainty, 2),
                                    "confidence_label": self._confidence_label(confidence),
                                    "explanation": explanation,
                                }
                            )

        if not combinations:
            return pd.DataFrame()

        result = pd.DataFrame(combinations).sort_values("total_score", ascending=False)
        result = result.head(20).reset_index(drop=True)
        result = self._apply_context_constraints(result, context_constraints)
        return result.head(10).reset_index(drop=True)
