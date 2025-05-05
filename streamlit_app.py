# streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


API_URL = "http://localhost:8000"  

st.set_page_config(page_title="ML Сервис", layout="wide")
st.title("ML-Сервис (FastAPI + Streamlit)")

tabs = st.tabs([
    "Загрузка данных",
    "Анализ данных",
    "Обучение модели",
    "Информация о модели",
    "Предсказание"
])


with tabs[0]:
    st.header("Загрузка датасета")
    uploaded_file = st.file_uploader("Загрузите CSV с 4 числовыми признаками (как у Iris)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.dataframe(df)
        st.success("Файл успешно загружен")

with tabs[1]:
    st.header("Анализ (EDA)")
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("Статистика")
        st.write(df.describe())

        st.subheader("Матрица корреляций")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Сначала загрузите данные на первой вкладке.")

with tabs[2]:
    st.header("Обучение модели")
    model_choice = st.selectbox("Выберите модель", ["logistic", "random_forest"])

    if model_choice == "logistic":
        max_iter = st.slider("Макс. итераций", min_value=100, max_value=1000, value=200, step=50)
        params = {"max_iter": max_iter}
    else:
        n_estimators = st.slider("Количество деревьев", min_value=10, max_value=200, value=100, step=10)
        params = {"n_estimators": n_estimators}

    if st.button("Обучить модель"):
        response = requests.post(f"{API_URL}/fit", params={"name": model_choice, **params})
        if response.status_code == 200:
            st.success(response.json()["status"])
        else:
            st.error("Ошибка при обучении модели")

    st.subheader("Или обучить модель на своих данных")
    x_file = st.file_uploader("Загрузите CSV с признаками (X)", key="x_upload")
    y_file = st.file_uploader("Загрузите CSV с метками (y)", key="y_upload")
    custom_model_name = st.selectbox("Выберите тип модели", ["logistic", "random_forest"], key="custom_model")

    if x_file and y_file and st.button("Обучить пользовательскую модель"):
        X = pd.read_csv(x_file).values.tolist()
        y = pd.read_csv(y_file).values.flatten().tolist()
        response = requests.post(f"{API_URL}/fit_custom", json={"X": X, "y": y, "name": custom_model_name})
        if response.status_code == 200:
            result = response.json()
            st.success(f"Модель обучена! Точность: {result['accuracy']:.2%}")
        else:
            st.error(f"Ошибка при обучении: {response.json()['detail']}")

with tabs[3]:
    st.header("Информация о модели")

    response = requests.get(f"{API_URL}/models")
    if response.status_code == 200:
        models = response.json()

        st.write("Текущая модель:", models["current_model"])
        st.write("Доступные модели:", models["available_models"])

        acc = models.get("accuracy")
        if acc is not None:
            st.metric(label="Точность модели", value=f"{acc:.2%}")

            fig, ax = plt.subplots()
            ax.bar(["Точность"], [acc], color='skyblue')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Доля правильных предсказаний")
            ax.set_title("Реальная точность модели на Iris")
            st.pyplot(fig)
        else:
            st.warning("Точность модели пока недоступна. Переобучите модель.")
    else:
        st.error("Ошибка при получении информации о модели")


with tabs[4]:
    st.header("Предсказание")
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.dataframe(df)
        if st.button("Сделать предсказание"):
            data = df.values.tolist()
            response = requests.post(f"{API_URL}/predict", json={"data": data})
            if response.status_code == 200:
                preds = response.json()["predictions"]
                df["prediction"] = preds
                st.dataframe(df)
            else:
                st.error("Ошибка предсказания")
    else:
        st.warning("Сначала загрузите данные.")
