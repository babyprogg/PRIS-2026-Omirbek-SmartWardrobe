import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from cv_engine import CVEngine
from data_manager import InventoryManager
from feedback import FeedbackLogger
from logic import StyleExpert
from nlp_engine import NLPEngine
from recommender import RecommendationEngine

st.set_page_config(page_title="Smart Wardrobe", layout="wide")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = DATA_DIR / "images"
DATA_PATH = RAW_DIR / "inventory.csv"
DATA_FEEDBACK_PATH = RAW_DIR / "feedback_events.csv"

MAX_VIBE_LENGTH = 280
MAX_COMBINATIONS = 1200


@st.cache_resource
def load_systems():
    manager = InventoryManager(str(DATA_PATH))
    expert = StyleExpert()
    nlp = NLPEngine()
    cv = CVEngine()
    rec = RecommendationEngine()
    feedback_logger = FeedbackLogger(str(DATA_FEEDBACK_PATH))
    rec.build_index(manager.data)
    return manager, expert, nlp, cv, rec, feedback_logger


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,1,0&display=swap');

        :root {
            --bg: #f2efe7;
            --surface: #fffdf8;
            --ink: #1f2937;
            --brand: #0f766e;
            --brand-2: #ea580c;
            --muted: #6b7280;
        }

        .stApp {
            background:
                radial-gradient(1200px 350px at 10% -20%, rgba(15, 118, 110, 0.15), transparent),
                radial-gradient(900px 300px at 95% 0%, rgba(234, 88, 12, 0.12), transparent),
                var(--bg);
            color: var(--ink);
            font-family: 'Manrope', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.02em;
            margin: 0;
        }

        .hero {
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            background: linear-gradient(130deg, #134e4a 0%, #115e59 40%, #7c2d12 100%);
            color: #ffffff;
            box-shadow: 0 18px 40px rgba(17, 24, 39, 0.2);
        }

        .hero .small {
            opacity: 0.88;
            font-weight: 600;
            font-size: 0.95rem;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid rgba(15, 118, 110, 0.12);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.07);
        }

        .metric-card .k {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-card .v {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .item-card {
            background: var(--surface);
            border: 1px solid rgba(15, 118, 110, 0.12);
            border-radius: 12px;
            padding: 0.75rem;
            min-height: 150px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .item-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(2, 6, 23, 0.08);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f5ef 0%, #f2efe7 100%);
            border-right: 1px solid rgba(15, 118, 110, 0.12);
        }

        section[data-testid="stSidebar"] .nav-link {
            border-radius: 10px !important;
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] .nav-link.active {
            background: linear-gradient(120deg, #0f766e 0%, #14b8a6 100%) !important;
            color: #fff !important;
            font-weight: 700;
        }

        .stButton > button {
            border-radius: 10px !important;
            border: 1px solid rgba(15, 118, 110, 0.35) !important;
        }

        .icon-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.2rem 0 0.65rem 0;
        }

        .icon-row .material-symbols-rounded {
            color: var(--brand);
            font-size: 1.45rem;
            line-height: 1;
        }

        .icon-row h2, .icon-row h3 {
            margin: 0;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def icon_heading(icon: str, text: str, level: int = 2):
    level = 2 if level not in (2, 3) else level
    st.markdown(
        f"""
        <div class='icon-row'>
            <span class='material-symbols-rounded'>{icon}</span>
            <h{level}>{text}</h{level}>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(analytics: dict):
    top_style = max(analytics["style_counts"], key=analytics["style_counts"].get) if analytics["style_counts"] else "-"
    top_color = max(analytics["color_counts"], key=analytics["color_counts"].get) if analytics["color_counts"] else "-"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='k'>Всего вещей</div>
                <div class='v'>{analytics['items_total']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='k'>Доминирующий стиль</div>
                <div class='v'>{top_style}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='k'>Частый цвет</div>
                <div class='v'>{top_color}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def save_uploaded_image(uploaded_file, prefix="item"):
    if uploaded_file is None:
        return ""

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix.lower() or ".jpg"
    file_name = f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"
    path = IMAGE_DIR / file_name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path.as_posix()


def _safe_image(path):
    p = str(path or "").strip()
    return p if p and os.path.exists(p) else ""


def _lookup_item(name, frame):
    if not name or name == "-" or frame.empty:
        return None
    rows = frame[frame["name"] == name]
    if rows.empty:
        return None
    return rows.iloc[0]


def render_item_preview(label, item_name, frame):
    item = _lookup_item(item_name, frame)
    st.markdown(f"**{label}**")
    if item is None:
        st.info("Не выбран")
        return

    image_path = _safe_image(item.get("image_path", ""))
    if image_path:
        st.image(image_path, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div class='item-card'>
                <div><strong>{item['name']}</strong></div>
                <div>Фото не добавлено</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(f"{item['category']} | {item['color']} | {item['style']}")


def category_gaps(frame):
    required = ["Top", "Bottom", "Shoes", "Accessory"]
    present = set(frame.get("category", pd.Series(dtype=str)).astype(str))
    return [cat for cat in required if cat not in present]


def estimate_outfit_combinations(frame, top_k_each, include_outerwear, include_shoes, include_accessory, rec_engine):
    if frame is None or frame.empty:
        return 0

    counts = frame.get("category", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    tops_pool = frame[frame.get("category", "") == "Top"]
    inner_tops, outerwear = rec_engine._split_tops(tops_pool)
    if inner_tops.empty:
        inner_tops = tops_pool
        outerwear = outerwear.iloc[0:0]

    tops = min(int(len(inner_tops)), top_k_each)
    bottoms = min(int(counts.get("Bottom", 0)), top_k_each)
    if tops == 0 or bottoms == 0:
        return 0

    outerwear_count = min(int(len(outerwear)), top_k_each)
    shoes_count = int(counts.get("Shoes", 0))
    accessory_count = int(counts.get("Accessory", 0))

    outerwear_mult = 1 if (not include_outerwear or outerwear_count == 0) else outerwear_count
    shoes_mult = 1 if (not include_shoes or shoes_count == 0) else min(shoes_count, top_k_each)
    accessory_mult = 1 if (not include_accessory or accessory_count == 0) else min(accessory_count, top_k_each)
    return tops * bottoms * outerwear_mult * shoes_mult * accessory_mult


def get_weather_notes(best_row, weather_c, frame, rec_engine):
    notes = []
    for part in ["top", "bottom", "outerwear", "shoes", "accessory"]:
        item = _lookup_item(best_row.get(part, ""), frame)
        if item is None:
            continue
        material = str(item.get("material", ""))
        note = rec_engine.explain_weather(material, weather_c)
        notes.append(f"{part.capitalize()} ({item['name']}): {note}")
    return notes


manager, expert, nlp, cv, rec, feedback_logger = load_systems()
inject_styles()

st.markdown(
    """
    <div class='hero'>
        <div class='small'>Smart Wardrobe</div>
        <h2 style='margin: 0.2rem 0 0.4rem 0;'>Персональный помощник для осмысленного подбора образов</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

analytics = rec.wardrobe_analytics()
fb = feedback_logger.summarize_feedback()

st.sidebar.header("Навигация")
with st.sidebar:
    page = option_menu(
        menu_title="Раздел",
        options=["Dashboard", "Outfit Studio", "Wardrobe", "Vision Lab"],
        icons=["grid", "stars", "bag", "camera"],
        menu_icon="layers",
        default_index=0,
    )


if page == "Dashboard":
    icon_heading("dashboard", "Dashboard")
    render_metrics(analytics)

    icon_heading("insights", "Инсайты гардероба", level=3)
    gaps = category_gaps(manager.data)
    if gaps:
        st.warning(f"Для более стабильных образов не хватает категорий: {', '.join(gaps)}")
    else:
        st.success("Базовое покрытие категорий полное: можно собирать образы с хорошей вариативностью")

    st.info(f"Фидбек пользователей: {fb['events_total']} оценок, положительных {fb['positive_rate']}%")

    c1, c2 = st.columns(2)
    with c1:
        icon_heading("styler", "Стили", level=3)
        style_df = pd.DataFrame(list(analytics["style_counts"].items()), columns=["style", "count"])
        if not style_df.empty:
            st.bar_chart(style_df.set_index("style"))
        else:
            st.caption("Пока нет данных")
    with c2:
        icon_heading("category", "Категории", level=3)
        cat_df = pd.DataFrame(list(analytics["category_counts"].items()), columns=["category", "count"])
        if not cat_df.empty:
            st.bar_chart(cat_df.set_index("category"))
        else:
            st.caption("Пока нет данных")


elif page == "Outfit Studio":
    icon_heading("checkroom", "Outfit Studio")
    st.caption("Генерация образов с учетом стиля, погоды, правил сочетаемости и пользовательского фидбека")

    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col1:
        occasion = st.selectbox("Повод", ["Работа", "Прогулка", "Свадьба", "Тренировка", "Свидание"])
        vibe = st.text_area(
            "Контекст",
            placeholder="Например: smart casual, аккуратно, без перегруза",
            max_chars=MAX_VIBE_LENGTH,
            help="Короткое описание дает более стабильный и понятный результат.",
        )
    with col2:
        weather_c = st.slider("Температура, °C", -20, 40, 18)
        include_outerwear = st.checkbox("Включать верхнюю одежду", value=True)
        include_shoes = st.checkbox("Включать обувь", value=True)
        include_accessory = st.checkbox("Включать аксессуары", value=True)
    with col3:
        top_k_each = st.slider(
            "Кандидатов на категорию",
            3,
            10,
            6,
            help="Сколько вещей берется в shortlist на каждую категорию перед сборкой комбинаций",
        )
        st.metric("Вещей в гардеробе", analytics["items_total"])

    estimated_combinations = estimate_outfit_combinations(
        manager.data,
        top_k_each,
        include_outerwear,
        include_shoes,
        include_accessory,
        rec,
    )
    st.caption(f"Оценка перебора комбинаций: {estimated_combinations}")
    if estimated_combinations > MAX_COMBINATIONS:
        st.warning(
            f"Перебор слишком большой ({estimated_combinations}). Система автоматически снизит глубину поиска для скорости."
        )

    if st.button("Сгенерировать образы"):
        effective_top_k = top_k_each
        effective_combinations = estimated_combinations
        while effective_combinations > MAX_COMBINATIONS and effective_top_k > 3:
            effective_top_k -= 1
            effective_combinations = estimate_outfit_combinations(
                manager.data,
                effective_top_k,
                include_outerwear,
                include_shoes,
                include_accessory,
                rec,
            )

        if effective_top_k != top_k_each:
            st.info(f"Для скорости и стабильности top_k автоматически уменьшен до {effective_top_k}")

        default_style = {
            "Работа": "Formal",
            "Прогулка": "Casual",
            "Свадьба": "Formal",
            "Тренировка": "Sport",
            "Свидание": "Smart Casual",
        }.get(occasion, "Casual")

        with st.spinner("Генерирую образы..."):
            nlp_output = nlp.analyze_text(vibe) if vibe.strip() else {
                "intent": default_style,
                "confidence": 1.0,
                "matched_keywords": [],
                "intent_distribution": {default_style: 1.0},
                "constraints": {},
            }
            style_from_text = nlp_output["intent"]
            context_constraints = nlp_output.get("constraints", {})

            effective_weather_c = weather_c
            if context_constraints.get("temperature_hint") is not None:
                effective_weather_c = int(context_constraints["temperature_hint"])

            preference_profile = rec.build_preference_profile(feedback_logger.data)

            outfits = rec.generate_outfits(
                target_style=style_from_text,
                weather_c=effective_weather_c,
                color_rules=expert.rules.get("color_matches", {}),
                top_k_each=effective_top_k,
                include_outerwear=include_outerwear,
                include_shoes=include_shoes,
                include_accessory=include_accessory,
                preference_profile=preference_profile,
                context_constraints=context_constraints,
            )

        if context_constraints.get("temperature_hint") is not None:
            st.caption(f"Использована температура из текста: {effective_weather_c}°C")

        st.info(f"Стиль: {style_from_text} | NLP confidence: {nlp_output['confidence']}")

        if outfits.empty:
            st.warning("Не удалось собрать образ. Проверьте наличие Top и Bottom в гардеробе")
        else:
            icon_heading("workspace_premium", "Топ комбинации", level=3)
            st.dataframe(outfits, use_container_width=True, hide_index=True)

            best = outfits.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Total", f"{best['total_score']}")
            with m2:
                st.metric("Style", f"{best['style_score']}")
            with m3:
                st.metric("Color", f"{best['color_score']}")
            with m4:
                st.metric("Weather", f"{best['weather_score']}")
            with m5:
                st.metric("Preference", f"{best['preference_score']}")

            st.caption(f"Уверенность: {best['confidence']} | uncertainty: {best['uncertainty']} | уровень: {best['confidence_label']}")
            st.write(f"Почему: {best['explanation']}")

            icon_heading("thermostat", "Детали погодного учета", level=3)
            notes = get_weather_notes(best, effective_weather_c, manager.data, rec)
            for note in notes:
                st.write(f"- {note}")

            icon_heading("collections", "Визуальный превью", level=3)
            p1, p2, p3, p4, p5 = st.columns(5)
            with p1:
                render_item_preview("Top", best["top"], manager.data)
            with p2:
                render_item_preview("Bottom", best["bottom"], manager.data)
            with p3:
                render_item_preview("Outerwear", best.get("outerwear", "-"), manager.data)
            with p4:
                render_item_preview("Shoes", best["shoes"], manager.data)
            with p5:
                render_item_preview("Accessory", best["accessory"], manager.data)

            f1, f2 = st.columns(2)
            with f1:
                if st.button("Оценить: подходит", key="feedback_like"):
                    feedback_logger.log_outfit_feedback(
                        top=best["top"],
                        bottom=best["bottom"],
                        outerwear=best.get("outerwear", "-"),
                        shoes=best["shoes"],
                        accessory=best["accessory"],
                        target_style=style_from_text,
                        total_score=best["total_score"],
                        feedback="like",
                    )
                    st.success("Оценка сохранена")
            with f2:
                if st.button("Оценить: не подходит", key="feedback_dislike"):
                    feedback_logger.log_outfit_feedback(
                        top=best["top"],
                        bottom=best["bottom"],
                        outerwear=best.get("outerwear", "-"),
                        shoes=best["shoes"],
                        accessory=best["accessory"],
                        target_style=style_from_text,
                        total_score=best["total_score"],
                        feedback="dislike",
                    )
                    st.warning("Оценка сохранена")


elif page == "Wardrobe":
    icon_heading("inventory_2", "Wardrobe")

    f1, f2, f3 = st.columns(3)
    with f1:
        cat_filter = st.selectbox("Категория", ["Все"] + sorted(manager.data["category"].dropna().unique().tolist()))
    with f2:
        style_filter = st.selectbox("Стиль", ["Все"] + sorted(manager.data["style"].dropna().unique().tolist()))
    with f3:
        view_mode = st.selectbox("Режим", ["Галерея", "Таблица"])

    filtered = manager.data.copy()
    if cat_filter != "Все":
        filtered = filtered[filtered["category"] == cat_filter]
    if style_filter != "Все":
        filtered = filtered[filtered["style"] == style_filter]

    if view_mode == "Таблица":
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        if filtered.empty:
            st.info("Нет вещей под текущие фильтры")
        else:
            cols = st.columns(4)
            for idx, (_, row) in enumerate(filtered.iterrows()):
                with cols[idx % 4]:
                    img = _safe_image(row.get("image_path", ""))
                    if img:
                        st.image(img, use_container_width=True)
                    st.markdown(f"**{row['name']}**")
                    st.caption(f"{row['category']} | {row['color']} | {row['style']}")

    st.divider()
    icon_heading("add_photo_alternate", "Добавить новую вещь", level=3)
    with st.form("add_item_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Название")
            cat = st.selectbox("Категория", ["Top", "Bottom", "Shoes", "Accessory"])
        with c2:
            clr = st.selectbox("Цвет", ["Белый", "Черный", "Синий", "Бежевый", "Серый", "Красный", "Зеленый"])
            material = st.selectbox("Материал", ["Хлопок", "Шерсть", "Деним", "Кожа", "Полиэстер", "Холст", "Шёлк", "Лен"])
        with c3:
            style_val = st.selectbox("Стиль", ["Casual", "Formal", "Sport", "Smart Casual"])
            photo = st.file_uploader("Фото вещи", type=["jpg", "jpeg", "png"], key="wardrobe_photo")

        submit = st.form_submit_button("Сохранить")
        if submit:
            if not name.strip():
                st.warning("Введите название вещи")
            else:
                image_path = save_uploaded_image(photo, prefix="wardrobe") if photo is not None else ""
                new_id = int(manager.data["id"].max()) + 1 if not manager.data.empty else 1
                added = manager.add_item(
                    {
                        "id": new_id,
                        "name": name.strip(),
                        "category": cat,
                        "color": clr,
                        "material": material,
                        "style": style_val,
                        "image_path": image_path,
                    }
                )
                if not added:
                    st.warning("Такая вещь уже есть в гардеробе")
                else:
                    rec.build_index(manager.data)
                    st.success("Вещь добавлена")
                    st.rerun()

    st.divider()
    icon_heading("delete", "Удалить вещь", level=3)
    if manager.data.empty:
        st.info("Гардероб пуст")
    else:
        options = {
            f"{int(row['id'])} | {row['name']} | {row['category']}": int(row['id'])
            for _, row in manager.data.sort_values("id").iterrows()
        }
        selected = st.selectbox("Выберите вещь", list(options.keys()))
        confirm_delete = st.checkbox("Подтверждаю удаление")
        if st.button("Удалить выбранную вещь"):
            if not confirm_delete:
                st.warning("Подтвердите удаление")
            else:
                removed = manager.delete_item_by_id(options[selected])
                if removed:
                    rec.build_index(manager.data)
                    st.success("Вещь удалена")
                    st.rerun()
                else:
                    st.warning("Не удалось удалить вещь")

    st.divider()
    icon_heading("tune", "Инструменты качества гардероба", level=3)

    q1, q2 = st.columns([2, 1])
    with q1:
        query = st.text_input("Сценарный поиск", placeholder="Например: базовый верх для smart casual")
    with q2:
        top_n = st.slider("Результатов", 1, 10, 5, key="wardrobe_top_n")

    if st.button("Подобрать по смыслу"):
        if query.strip():
            results = rec.find_similar(query, top_n=top_n)
            if results.empty:
                st.warning("Ничего не найдено")
            else:
                st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.warning("Введите запрос")

    threshold = st.slider("Порог похожести для дубликатов, %", 70, 95, 82, key="wardrobe_dup")
    if st.button("Найти дубликаты"):
        dup = rec.find_near_duplicates(threshold=threshold)
        if dup.empty:
            st.success("Явных дубликатов не найдено")
        else:
            st.dataframe(dup, use_container_width=True, hide_index=True)


elif page == "Vision Lab":
    icon_heading("photo_camera", "Vision Lab")
    st.caption("Визуальный анализ и поиск похожих вещей с приоритетом категории")

    uploaded_file = st.file_uploader("Загрузите фото вещи", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)

        if st.button("Проанализировать фото"):
            with st.spinner("Анализ изображения"):
                st.session_state["vision_result"] = cv.process_image(uploaded_file.getvalue())

        result = st.session_state.get("vision_result")
        if result:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Определенный цвет", result["color"])
                st.metric("Уверенность цвета", result.get("color_confidence", 0.0))
            with c2:
                vision_category = st.selectbox("Категория", ["Top", "Bottom", "Shoes", "Accessory"], index=0)
                vision_style = st.selectbox("Стиль", ["Casual", "Formal", "Sport", "Smart Casual"], index=0)

            st.caption("Поиск работает с приоритетом категории (вес 0.65), затем стиль (0.2) и цвет (0.15)")
            similar = rec.find_similar_structured(
                category=vision_category,
                color=result["color"],
                style=vision_style,
                top_n=5,
            )
            if similar.empty:
                st.info("Похожих вещей не найдено")
            else:
                st.dataframe(similar, use_container_width=True, hide_index=True)

            st.divider()
            icon_heading("playlist_add", "Добавить в гардероб из Vision Lab", level=3)
            add_name = st.text_input("Название новой вещи", value="Новая вещь", key="vision_add_name")
            add_material = st.selectbox(
                "Материал",
                ["Хлопок", "Шерсть", "Деним", "Кожа", "Полиэстер", "Холст", "Шёлк", "Лен"],
                key="vision_material",
            )
            if st.button("Добавить вещь в гардероб"):
                if not add_name.strip():
                    st.warning("Введите название")
                else:
                    image_path = save_uploaded_image(uploaded_file, prefix="vision")
                    new_id = int(manager.data["id"].max()) + 1 if not manager.data.empty else 1
                    added = manager.add_item(
                        {
                            "id": new_id,
                            "name": add_name.strip(),
                            "category": vision_category,
                            "color": result["color"],
                            "material": add_material,
                            "style": vision_style,
                            "image_path": image_path,
                        }
                    )
                    if not added:
                        st.warning("Такая вещь уже есть в гардеробе")
                    else:
                        rec.build_index(manager.data)
                        st.success("Вещь добавлена в гардероб")
                        st.rerun()
        elif "vision_result" in st.session_state and st.session_state["vision_result"] is None:
            st.error("Не удалось обработать изображение. Попробуйте другой файл.")
