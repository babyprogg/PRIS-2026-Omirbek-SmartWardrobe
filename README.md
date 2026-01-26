# Smart Wardrobe AI

[cite_start]**Дисциплина:** Проектирование и разработка интеллектуальных систем [cite: 2]

## Описание проекта

Интеллектуальный помощник для управления гардеробом. [cite_start]Система помогает классифицировать одежду по фотографиям и подбирать образы на основе экспертных правил стиля и повода (дресс-кода). [cite: 226]

## Стек технологий

- [cite_start]**Язык:** Python 3.11+ [cite: 162]
- [cite_start]**Интерфейс:** Streamlit [cite: 75, 208]
- [cite_start]**Интеллект:** OpenCV, Transformers (HuggingFace), Rule-based logic [cite: 162, 201]

## Архитектура системы

````mermaid
graph TD;
    User[Пользователь] -->|Фото одежды| UI[Streamlit App];
    UI -->|Изображение| CV[CV Модуль: Классификация];
    CV -->|Тип: Рубашка/Брюки| Logic[Экспертная система: Правила сочетания];
    UI -->|Контекст: Свадьба/Работа| NLP[NLP Модуль: Анализ повода];
    NLP -->|Стиль: Formal| Logic;
    Logic -->|Вердикт: Рекомендация лука| UI;
[cite_start]``` [cite: 96, 114]

## Установка
1. `python -m venv venv`
2. `.\venv\Scripts\activate`
3. `pip install -r requirements.txt` [cite: 48, 78]
````
