#  ML-Сервис 
Этот проект представляет собой простой ML-сервис с веб-интерфейсом на Streamlit и HTTP API на FastAPI. Сервис поддерживает обучение, предсказание, выбор модели и отображение метрик.
Рекомендуем использовать для тестирования файл iris.csv(Приложен)
---

##  Быстрый запуск 

```bash
git clone https://github.com/Rodinkufa/ml_service.git
cd ml_service
docker-compose up --build

FastAPI доступен по адресу: http://localhost:8000/docs

Streamlit доступен по адресу: http://localhost:8501

