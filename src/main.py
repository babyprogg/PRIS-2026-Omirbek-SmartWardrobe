import streamlit as st
from data_manager import InventoryManager
from logic import Garment, StyleExpert
from nlp_engine import NLPEngine
from cv_engine import CVEngine

# 1. Инициализация систем
data_path = "data/raw/inventory.csv"
manager = InventoryManager(data_path)
expert = StyleExpert()
nlp = NLPEngine()
cv = CVEngine()

# Настройка страницы
st.set_page_config(page_title="AI Smart Wardrobe", page_icon="👔")
st.title("AI Smart Wardrobe 👔")

# 2. Единое боковое меню (Навигация)
st.sidebar.header("Меню управления")
page = st.sidebar.selectbox("Выберите раздел:", 
    ["Главная", "Мой Шкаф", "Подбор образа", "Загрузка одежды"]
)

# --- СТРАНИЦА: ГЛАВНАЯ ---
if page == "Главная":
    st.write("### Добро пожаловать в ваш интеллектуальный стилист!")
    
    st.header("💬 AI Ассистент (NLP)")
    user_input = st.text_input("Опишите ваши планы (например: 'Иду на бизнес-встречу' или 'Хочу погулять в парке')")
    
    if user_input:
        intent = nlp.extract_intent(user_input)
        st.info(f"🤖 Система определила стиль: **{intent}**")
        # Сопоставляем интент с рекомендациями экспертной системы
        mapping = {"Formal": "Работа", "Casual": "Прогулка", "Sport": "Прогулка"}
        rec = expert.get_recommendation(mapping.get(intent, "Прогулка"))
        st.success(rec)

    if st.button("Проверить готовность систем"):
        st.success("Все модули (NLP, CV, Logic, Data) инициализированы! 🚀")

# --- СТРАНИЦА: МОЙ ШКАФ ---
elif page == "Мой Шкаф":
    st.header("📦 Содержимое вашего гардероба")
    st.dataframe(manager.data, use_container_width=True)
    
    st.subheader("➕ Добавить новую вещь")
    with st.form("add_item_form"):
        name = st.text_input("Название (например: Белая рубашка)")
        cat = st.selectbox("Категория", ["Top", "Bottom", "Shoes"])
        clr = st.selectbox("Цвет", ["Белый", "Черный", "Синий", "Бежевый", "Красный"])
        submit = st.form_submit_button("Сохранить в базу")
        
        if submit:
            new_id = len(manager.data) + 1
            new_item = {"id": new_id, "name": name, "category": cat, "color": clr}
            manager.add_item(new_item)
            st.success(f"Вещь '{name}' добавлена!")
            st.rerun()

# --- СТРАНИЦА: ПОДБОР ОБРАЗА (Экспертная система) ---
elif page == "Подбор образа":
    st.header("🧠 Экспертный подбор сочетаний")
    
    col1, col2 = st.columns(2)
    with col1:
        top_color = st.selectbox("Цвет верха", ["Белый", "Черный", "Синий", "Бежевый"])
        # top_item = Garment("Верх", "Top", top_color)
    with col2:
        bottom_color = st.selectbox("Цвет низа", ["Черный", "Белый", "Синий", "Бежевый"])
        # bottom_item = Garment("Низ", "Bottom", bottom_color)
    
    if st.button("Проверить сочетание"):
        item1 = Garment("Верх", "Top", top_color)
        item2 = Garment("Низ", "Bottom", bottom_color)
        is_ok, message = expert.check_combination(item1, item2)
        if is_ok:
            st.success(message)
        else:
            st.warning(message)

# --- СТРАНИЦА: ЗАГРУЗКА ОДЕЖДЫ ---
elif page == "Загрузка одежды":
    st.header("📸 Распознавание вещей")
    uploaded_file = st.file_uploader("Загрузите фото вещи для анализа...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Загруженное изображение', use_column_width=True)
        
        if st.button("🚀 Начать анализ изображения"):
            with st.spinner('Нейросеть анализирует изображение...'):
                img_bytes = uploaded_file.getvalue()
                result = cv.process_image(img_bytes)
                
                if result:
                    st.success("Анализ завершен!")
                    st.write(f"🎨 **Определенный цвет**: {result['color']}")
                    
                    if result['text']:
                        st.write(f"📝 **Найденный текст на бирках**: {', '.join(result['text'])}")
                    
                    st.info("💡 Вы можете сохранить эту вещь в гардероб, подтвердив данные в разделе 'Мой Шкаф'.")
                else:
                    st.error("Ошибка при обработке изображения.")