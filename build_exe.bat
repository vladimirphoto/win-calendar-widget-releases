@echo off
rem Сборка Lumina Retouch в один .exe (Windows). Запускать из этой папки.
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --windowed --name "Lumina Retouch" app.py
echo.
echo Готово: dist\Lumina Retouch\Lumina Retouch.exe
pause
