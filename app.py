"""
MVP: Streamlit-демо классификатора типов математических ошибок
Магистерская диссертация Мелконяна О. А.

Запуск:
    pip install -r requirements.txt
    streamlit run app.py

Сначала запустить MVP_Colab_Notebook.ipynb для генерации best_classifier.joblib
и recommendations.json. Эти файлы должны лежать в той же папке, что и app.py.
"""

import streamlit as st
import joblib
import json
import time
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Диагностика математических ошибок",
    page_icon="🧮",
    layout="wide"
)


@st.cache_resource
def load_model_and_classifier():
    """Загрузка модели e5-large и обученного классификатора. Кэшируется в памяти процесса."""
    embedder = SentenceTransformer('intfloat/multilingual-e5-large')
    classifier = joblib.load('best_classifier.joblib')
    with open('recommendations.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return embedder, classifier, meta['recommendations'], meta['user_facing_labels']


def diagnose(problem, solution, embedder, classifier, recs, labels):
    """Запуск пайплайна на одной паре условие+решение."""
    full_text = f'query: {problem} Решение: {solution}'

    t0 = time.time()
    emb = embedder.encode([full_text], convert_to_numpy=True)
    embed_time = (time.time() - t0) * 1000

    t1 = time.time()
    predicted = classifier.predict(emb)[0]
    proba = classifier.predict_proba(emb)[0]
    confidence = float(proba.max())
    classify_time = (time.time() - t1) * 1000

    return {
        'label_internal': predicted,
        'label_user': labels[predicted],
        'confidence': round(confidence, 3),
        'recommendation': recs[predicted],
        'embed_time_ms': round(embed_time, 1),
        'classify_time_ms': round(classify_time, 1),
        'total_time_ms': round(embed_time + classify_time, 1),
        'needs_llm_escalation': confidence < 0.8
    }


# Интерфейс
st.title("🧮 Диагностика математических ошибок")
st.markdown(
    "**MVP магистерской диссертации.** Автоматическая классификация типа ошибки "
    "в ученическом решении задачи по математике 5–9 классов через embedding-based ML."
)

# Загружаем модели
with st.spinner("Загрузка модели e5-large (90 секунд при первом запуске)..."):
    embedder, classifier, recs, labels = load_model_and_classifier()
st.success("Модель загружена.")

# Колонки для ввода
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Ввод данных")

    fgos_section = st.selectbox(
        "Раздел ФГОС ООО",
        options=["дроби", "уравнения", "степени", "геометрия", "арифметика", "другое"],
        index=0
    )

    problem = st.text_area(
        "Условие задачи",
        height=80,
        placeholder="Например. Найдите сумму дробей 1/3 + 1/4"
    )

    solution = st.text_area(
        "Решение школьника",
        height=120,
        placeholder="Например. 1/3 + 1/4 = 2/7"
    )

    # Готовые примеры
    st.markdown("**Готовые примеры для демонстрации**")
    example_choice = st.radio(
        "Выбрать пример",
        options=[
            "—",
            "Концептуальная (дроби)",
            "Вычислительная (уравнение)",
            "Процедурная (квадратное уравнение)",
            "Невнимательность (геометрия)"
        ],
        horizontal=False
    )

    examples = {
        "Концептуальная (дроби)": (
            "Найдите сумму дробей 1/3 + 1/4",
            "1/3 + 1/4 = 2/7"
        ),
        "Вычислительная (уравнение)": (
            "Решите уравнение 3x + 7 = 22",
            "3x = 22 - 7. 3x = 14. x = 14/3"
        ),
        "Процедурная (квадратное уравнение)": (
            "Решите уравнение x^2 = 9",
            "x = корень из 9. x = 3"
        ),
        "Невнимательность (геометрия)": (
            "Найдите периметр прямоугольника со сторонами 3 и 5 см",
            "P = 3 * 5 = 15 см^2"
        )
    }

    if example_choice != "—":
        problem_demo, solution_demo = examples[example_choice]
        problem = st.text_area("Пример условия", value=problem_demo, height=80, key="ex_p")
        solution = st.text_area("Пример решения", value=solution_demo, height=120, key="ex_s")

    run_btn = st.button("Классифицировать ошибку", type="primary", use_container_width=True)

with col_output:
    st.subheader("Результат диагностики")

    if run_btn and problem and solution:
        with st.spinner("Векторизация и классификация..."):
            result = diagnose(problem, solution, embedder, classifier, recs, labels)

        # Главный диагноз
        st.markdown(f"### 🔍 {result['label_user']}")

        # Confidence с цветовой раскраской
        conf_color = "green" if result['confidence'] >= 0.8 else ("orange" if result['confidence'] >= 0.6 else "red")
        st.markdown(
            f"**Уверенность.** <span style='color:{conf_color}; font-size:20px'>{result['confidence']:.2f}</span>",
            unsafe_allow_html=True
        )

        if result['needs_llm_escalation']:
            st.warning(
                "⚠️ Низкая confidence (<0.8). В production-режиме здесь произошла бы эскалация "
                "к LLM-fallback (GigaChat или GPT-4o) для развёрнутого объяснения."
            )

        # Рекомендация
        st.markdown("**Рекомендация педагогической реакции.**")
        st.info(result['recommendation'])

        # Метрики latency
        st.markdown("---")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Векторизация", f"{result['embed_time_ms']} мс")
        col_m2.metric("Классификация", f"{result['classify_time_ms']} мс")
        col_m3.metric("Всего", f"{result['total_time_ms']} мс")

    elif run_btn:
        st.error("Заполните условие и решение перед классификацией.")
    else:
        st.markdown("_Введите условие и решение, затем нажмите кнопку слева._")

# Подвал
st.markdown("---")
st.caption(
    "Магистерская диссертация Мелконяна О. А. | "
    "Программа «Электронный бизнес и цифровые инновации», ВШЭ | "
    "embedding-based ML на multilingual-e5-large + sklearn"
)
