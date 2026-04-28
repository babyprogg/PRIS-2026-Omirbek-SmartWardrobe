# Smart Wardrobe AI

Интеллектуальный ассистент гардероба на Python + Streamlit.

Проект сочетает:
- NLP-анализ пользовательского запроса (Natasha + гибридный semantic scoring)
- Rule-based и score-based генерацию образов
- Vision Lab для определения цвета и поиска похожих вещей
- Персонализацию через explicit feedback (like/dislike)

## Архитектура

- `src/main.py` — интерфейс Streamlit и пользовательские сценарии
- `src/nlp_engine.py` — извлечение намерения, confidence и контекстных ограничений
- `src/recommender.py` — генерация и ранжирование образов, explainability, weather-fit
- `src/data_manager.py` — хранение и нормализация инвентаря
- `src/cv_engine.py` — визуальный модуль определения доминирующего цвета
- `src/feedback.py` — логирование и агрегирование обратной связи
- `src/evaluation.py` — быстрая оценка качества для демонстрации

## Данные

- `data/raw/inventory.csv` — предметы гардероба
- `data/raw/rules.json` — правила сочетаемости/ключевые слова
- `data/raw/feedback_events.csv` — события обратной связи
- `data/images/` — изображения вещей

## Быстрый запуск (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src/main.py
```

## Проверка качества

```powershell
python src/evaluation.py
```

Ожидается метрики вида:
- `generation_success_rate`
- `avg_total_score`
- `avg_confidence`
- `top_diversity_ratio`

## Что уже реализовано

- Контекстный подбор: стиль + погода + цветовые правила + пользовательские предпочтения
- Ограничения из текста: предпочитаемые/запрещенные цвета и материалы, температурный hint
- Визуальное добавление вещей в гардероб
- Поиск похожих вещей и детекция near-duplicates
- Explainable scoring (style/color/weather/preference)

## Ограничения текущей версии

- CV-модуль определяет только доминирующий цвет (без полноценной классификации одежды)
- NLP работает без тяжелых LLM-моделей, поэтому редкие формулировки могут трактоваться неоднозначно
- Персонализация лучше работает после накопления feedback-событий

## Быстрый roadmap

1. Добавить небольшой тестовый набор для ключевых функций (`pytest`)
2. Добавить экспорт top-outfit в PDF/PNG для презентации
3. Расширить CV до классификации категории (Top/Bottom/Shoes/Accessory)
