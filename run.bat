@echo off
rem Первый запуск: ставит зависимости и открывает приложение.
python -m pip install -r requirements.txt
python app.py %*
