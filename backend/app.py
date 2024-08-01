from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask_cors import CORS
from datetime import datetime, timedelta
import ccxt
import requests
import time
from snownlp import SnowNLP
from textblob import TextBlob

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 允許所有來源

NEWSAPI_KEY = "0160867f82e74d9daaa9b820e2052860"


# 格式化加密貨幣對的符號
def format_symbol(symbol):
    if symbol.upper().endswith("USD") and "/" not in symbol:
        return symbol.upper().replace("USD", "/USDT")
    return symbol.upper()


# 計算技術指標
def add_technical_indicators(data):
    # 移動平均線（MA）
    data["MA_20"] = data["Adj Close"].rolling(window=20).mean()
    data["MA_50"] = data["Adj Close"].rolling(window=50).mean()

    # 相對強弱指數（RSI）
    delta = data["Adj Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data["RSI"] = 100 - (100 / (1 + RS))

    # 移動平均收斂散度（MACD）
    EMA_30 = data["Adj Close"].ewm(span=30, adjust=False).mean()
    EMA_200 = data["Adj Close"].ewm(span=200, adjust=False).mean()
    data["MACD"] = EMA_30 - EMA_200
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # 填補缺失值
    data.fillna(0, inplace=True)
    return data


# 加載並預處理股票或加密貨幣資料
def load_stock_data(symbol, start, end):
    symbol = format_symbol(symbol)
    print(f"從 {start} 到 {end} 獲取 {symbol} 的資料")
    if "/" in symbol:  # 檢測加密貨幣對
        exchange = ccxt.binance(
            {"enableRateLimit": True, "verbose": False}  # 關閉詳細日誌
        )
        since = exchange.parse8601(f"{start}T00:00:00Z")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, "1d", since)
            if not ohlcv:
                print(f"未返回 {symbol} 的 OHLCV 資料")
                return pd.DataFrame()
            else:
                print(
                    f"獲取到 {symbol} 的 OHLCV 資料: {ohlcv[:5]}..."
                )  # 打印前5行資料進行調試
        except Exception as e:
            print(f"從 Binance 獲取資料時出錯: {e}")
            return pd.DataFrame()  # 返回空資料框，表示無法獲取資料

        data = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        data.set_index("timestamp", inplace=True)
        data = data[["close"]]
        data.columns = ["Adj Close"]
    else:
        # 股票資料
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            print(f"使用 yfinance 未返回 {symbol} 的資料")
        data = data[["Adj Close"]]

    print(f"從 {start} 到 {end} 加載 {symbol} 的資料: {data.shape[0]} 行")

    # 添加技術指標
    data = add_technical_indicators(data)

    return data


# 建立和訓練 LSTM 模型
def create_lstm_model(data, time_steps=80):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    print(f"縮放後的資料形狀: {scaled_data.shape}")

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    print(f"訓練資料形狀: {train_data.shape}")

    if len(train_data) < time_steps:
        raise ValueError(
            f"訓練資料不足: 只有 {len(train_data)} 行，但需要至少 {time_steps} 行。"
        )

    X_train, Y_train = [], []
    for i in range(time_steps, len(train_data)):
        X_train.append(train_data[i - time_steps : i])  # 這裡直接保留所有特徵
        Y_train.append(train_data[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    print(f"重新調整前的 X_train 形狀: {X_train.shape}")
    print(f"Y_train 形狀: {Y_train.shape}")

    # 修正 X_train 的形狀
    if len(X_train.shape) == 3:  # 確保形狀為 (樣本數, 時間步長, 特徵數)
        X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1], X_train.shape[2]
        )  # 保留特徵數

    print(f"重新調整後的 X_train 形狀: {X_train.shape}")

    # 確認 X_train 形狀是否正確
    if len(X_train.shape) != 3 or X_train.shape[1] != time_steps:
        raise ValueError(f"X_train 形狀不正確: {X_train.shape}")

    model = Sequential()
    model.add(
        LSTM(
            50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )  # 更新這裡的 input_shape
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, Y_train, epochs=5, batch_size=32)

    print("模型訓練完成")
    return model, scaler


# 使用 NewsAPI 獲取市場新聞資料
def fetch_market_news_for_symbol(symbol):
    try:
        # 使用 NewsAPI 查詢與符號相關的新聞
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        data = response.json()

        print(f"API 回應資料: {data}")  # 打印API回應的完整資料

        if "articles" in data:
            articles = data["articles"]
            return articles
        else:
            print(f"未能獲取資料或回應錯誤：{data.get('message', '無其他信息')}")
            return None
    except Exception as e:
        print(f"API請求過程中出錯：{e}")
        return None


# 提取新聞標題並進行情感分析
def analyze_news_sentiment(articles):
    sentiments = []
    for article in articles:
        title = article.get("title", "")
        sentiment = TextBlob(title).sentiment.polarity
        sentiments.append(sentiment)
    return sentiments


@app.route("/predict", methods=["GET"])
def predict():
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "需要提供股票代碼"}), 400

        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

        data = load_stock_data(symbol, start_date, end_date)
        if data.empty:
            return jsonify({"error": "未找到股票資料"}), 404

        if len(data) < 100:
            return jsonify({"error": f"{symbol} 資料不足"}), 400

        model, scaler = create_lstm_model(data.values)

        test_data = data[-100:].values
        test_data = scaler.transform(test_data)
        X_test = np.array([test_data] * 50).reshape(
            (50, test_data.shape[0], test_data.shape[1])
        )

        predictions = model.predict(X_test).flatten()

        predictions_scaled = [
            float(
                scaler.data_min_[0] + (scaler.data_max_[0] - scaler.data_min_[0]) * pred
            )
            for pred in predictions
        ]
        average_prediction = float(np.mean(predictions_scaled))

        last_close_price = float(data["Adj Close"].iloc[-1])
        if abs(average_prediction - last_close_price) > last_close_price * 0.15:
            average_prediction = last_close_price

        print(f"{symbol} 的平均預測結果: {average_prediction}")

        if average_prediction > last_close_price:
            trend = "看漲"
        else:
            trend = "看跌"

        print(f"{symbol} 的平均預測結果: {average_prediction}, 趨勢: {trend}")

        return jsonify(
            {"symbol": symbol, "prediction": average_prediction, "trend": trend}
        )
    except Exception as e:
        print(f"預測過程中出錯: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/market-sentiment", methods=["GET"])
def market_sentiment():
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "必須提供查詢詞"}), 400

        articles = fetch_market_news_for_symbol(symbol)
        if articles is None:
            return jsonify({"error": "無法獲取新聞資料"}), 500

        sentiments = analyze_news_sentiment(articles)
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        return jsonify(
            {
                "average_sentiment": average_sentiment,
                "trend": "看漲" if average_sentiment > 0 else "看跌",
            }
        )
    except Exception as e:
        print(f"市場情緒分析過程中出錯: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sentiment", methods=["GET"])
def sentiment():
    try:
        text = request.args.get("text")
        if not text:
            return jsonify({"error": "必須提供文本"}), 400

        # 使用 SnowNLP 進行中文情感分析
        analysis = SnowNLP(text)
        sentiment = analysis.sentiments  # 返回值範圍是0到1，1表示情緒更積極
        return jsonify({"text": text, "sentiment": sentiment})
    except Exception as e:
        print(f"情緒分析過程中出錯: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
