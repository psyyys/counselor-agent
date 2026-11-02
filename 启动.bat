@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist data\index\exemplars.pkl (
  echo 第一次运行：正在建检索库，需要几分钟……
  python -X utf8 -m counselor.build_index
)
start "" http://127.0.0.1:8765
python -X utf8 -m uvicorn counselor.server:app --host 127.0.0.1 --port 8765
pause
