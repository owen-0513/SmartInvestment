<template>
    <div id="app">
        <header class="App-header">
            <h1>投資分析</h1>
            <div>
                <input v-model="symbol" type="text" placeholder="輸入股價代碼">
                <button @click="getPrediction" :disabled="loading">預測</button>
            </div>
            <div v-if="prediction !== null">
                <p>股價預測 {{ symbol }}: {{ prediction }}</p>
                <p>趨勢: {{ trend }}</p>
            </div>
            <div v-if="errorMessage">
                <p style="color: red;">{{ errorMessage }}</p>
            </div>
            <div>
                <button @click="getSentiment" :disabled="loading">分析市場情緒</button>
            </div>
            <div v-if="sentiment !== null">
                <p>{{ symbol }} 市場情緒值: {{ sentiment.toFixed(3) }}</p>
                <p>{{ symbol }} 市場情緒趨勢: {{ sentimentTrend }}</p>
            </div>
            <div v-if="sentiment !== null" class="sentiment-explanation">
                <p>市場情緒值範圍說明：</p>
                <ul>
                    <li><strong>0 - 0.2:</strong> 市場情緒非常負面，投資者普遍悲觀。</li>
                    <li><strong>0.2 - 0.4:</strong> 市場情緒較為負面，存在一定的擔憂。</li>
                    <li><strong>0.4 - 0.6:</strong> 市場情緒中性，投資者觀望。</li>
                    <li><strong>0.6 - 0.8:</strong> 市場情緒較為正面，投資者信心逐漸增強。</li>
                    <li><strong>0.8 - 1:</strong> 市場情緒非常正面，投資者普遍樂觀。</li>
                </ul>
            </div>
            <div v-if="sentimentErrorMessage">
                <p style="color: red;">{{ sentimentErrorMessage }}</p>
            </div>
        </header>
    </div>
</template>

<script>
import axios from 'axios';

export default {
    name: 'App',
    data() {
        return {
            symbol: '',
            prediction: null,
            trend: '',
            sentiment: null,
            sentimentTrend: '',
            errorMessage: '',
            sentimentErrorMessage: '',
            loading: false,
        };
    },
    methods: {
        async getPrediction() {
            this.loading = true;
            this.errorMessage = '';
            try {
                const response = await axios.get(`http://localhost:5000/predict?symbol=${this.symbol}`);
                console.log("Prediction response:", response.data);
                this.prediction = response.data.prediction;
                this.trend = response.data.trend;
            } catch (error) {
                console.error("Error fetching prediction:", error);
                this.errorMessage = error.response && error.response.data ? error.response.data.error : "獲取預測時發生錯誤";
            } finally {
                this.loading = false;
            }
        },
        async getSentiment() {
            this.loading = true;
            this.sentimentErrorMessage = '';
            try {
                const response = await axios.get(`http://localhost:5000/market-sentiment?symbol=${this.symbol}`);
                console.log("Market Sentiment response:", response.data);
                this.sentiment = response.data.average_sentiment;
                this.sentimentTrend = response.data.trend;
            } catch (error) {
                console.error("Error fetching market sentiment:", error);
                this.sentimentErrorMessage = error.response && error.response.data ? error.response.data.error : "獲取市場情緒時發生錯誤";
            } finally {
                this.loading = false;
            }
        },
    },
};
</script>

<style>
#app {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    margin-top: 60px;
}

.App-header {
    background-color: #282c34;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: calc(10px + 2vmin);
    color: white;
}

input[type="text"] {
    padding: 10px;
    margin: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    width: 250px;
}

button {
    padding: 10px 20px;
    margin: 10px;
    border-radius: 5px;
    background-color: #4CAF50;
    color: white;
    border: none;
    cursor: pointer;
}

button:disabled {
    background-color: #888;
    cursor: not-allowed;
}

p {
    margin: 10px;
}

p[style="color: red;"] {
    color: red;
    font-weight: bold;
}

.sentiment-explanation {
    text-align: left;
    margin-top: 20px;
    background-color: #333;
    padding: 15px;
    border-radius: 5px;
}

.sentiment-explanation p {
    margin-bottom: 10px;
}

.sentiment-explanation ul {
    list-style-type: none;
    padding: 0;
}

.sentiment-explanation ul li {
    margin: 5px 0;
}

.sentiment-explanation ul li strong {
    color: #4CAF50;
}
</style>
