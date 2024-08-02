# 使用官方 Python 鏡像作為基礎鏡像
FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 複製 `requirements.txt` 並安裝依賴項
COPY backend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# 複製應用程式代碼到容器中
COPY backend/ .

# 設置環境變數
ENV PORT=8080

# 啟動 Flask 應用
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
