import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Bidirectional, GRU
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Function to fetch data from CoinGecko API
def fetch_data(crypto, vs_currency, days):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency={vs_currency}&days={days}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'prices' in data and 'total_volumes' in data:
            prices = data['prices']
            volumes = data['total_volumes']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['volume'] = [v[1] for v in volumes]
            return df
        else:
            st.error("API response does not contain 'prices' or 'total_volumes' key.")
            st.stop()
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        st.stop()

# Preprocess data
def preprocess_data(df):
    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    df['price'] = price_scaler.fit_transform(df[['price']])
    df['volume'] = volume_scaler.fit_transform(df[['volume']])
    return df, price_scaler, volume_scaler

# Create sequences for RNN, LSTM, Bi-LSTM, GRU
def create_sequences(df, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:i+n_steps].values)
        y.append(df.iloc[i+n_steps].values)
    return np.array(X), np.array(y)

# Build RNN model
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build Bi-LSTM model
def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build GRU model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to train and predict
def train_and_predict(model, X_train, y_train, X_test, n_steps):
    model.fit(X_train, y_train, epochs=100, verbose=0)
    test_predictions = model.predict(X_test)
    future_predictions = []
    current_input = X_test[-1]
    for _ in range(len(X_test)):
        pred = model.predict(current_input.reshape(1, n_steps, 1))
        future_predictions.append(pred[0][0])
        current_input = np.roll(current_input, -1)
        current_input[-1, 0] = pred
    return test_predictions, future_predictions

# Calculate TWAP
def calculate_twap(prices):
    return np.mean(prices)

# Calculate VWAP
def calculate_vwap(prices, volumes):
    return np.sum(prices * volumes) / np.sum(volumes)

# Fetch recent news articles
def fetch_news(crypto, api_key, language='en'):
    url = f'https://newsapi.org/v2/everything?q={crypto}&language={language}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        st.error(f"Failed to fetch news: {response.status_code}")
        return []

# Analyze sentiment
def analyze_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(article['title'])['compound'] for article in articles]
    return np.mean(sentiments)

# Streamlit app
st.title("Bitcoin Price Prediction with Sentiment Analysis")
crypto = st.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "litecoin"])
vs_currency = "usd"
days = st.slider("Days to Predict", 1, 365, 30)
model_choice = st.selectbox("Prediction Model", ["RNN", "LSTM", "Bi-LSTM", "GRU", "Random Forest", "Gradient Boosting", "XGBoost"])

# Fetch and preprocess data
crypto_data = fetch_data(crypto, vs_currency, 3*days)  # Fetch double the days for train and test split
crypto_data, price_scaler, volume_scaler = preprocess_data(crypto_data)

# Fetch and analyze news sentiment
api_key = "b4c68fd5199740869a7b0b269ce55c31"
articles = fetch_news(crypto, api_key)
sentiment = analyze_sentiment(articles)
st.write(f"Average Sentiment Score: {sentiment}-neutral")

# Prepare data for RNN, LSTM, Bi-LSTM, GRU
n_steps = days
X, y = create_sequences(crypto_data[['price']], n_steps)

# Split the data into 80% training and 20% testing
split_index = int(len(X) * 0.8)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Build and train models
if model_choice == "RNN":
    model = build_rnn_model((n_steps, 1))
    test_predictions, future_predictions = train_and_predict(model, X_train, y_train, X_test, n_steps)
elif model_choice == "LSTM":
    model = build_lstm_model((n_steps, 1))
    test_predictions, future_predictions = train_and_predict(model, X_train, y_train, X_test, n_steps)
elif model_choice == "Bi-LSTM":
    model = build_bilstm_model((n_steps, 1))
    test_predictions, future_predictions = train_and_predict(model, X_train, y_train, X_test, n_steps)
elif model_choice == "GRU":
    model = build_gru_model((n_steps, 1))
    test_predictions, future_predictions = train_and_predict(model, X_train, y_train, X_test, n_steps)
elif model_choice == "Random Forest":
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    test_predictions = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
    future_X = X_test[-1].reshape(1, -1, 1)
    future_predictions = []
    for _ in range(days):
        pred = rf_model.predict(future_X.reshape(1, -1))
        future_predictions.append(pred[0])
        future_X = np.roll(future_X, -1)
        future_X[0, -1, 0] = pred
elif model_choice == "Gradient Boosting":
    gb_model = GradientBoostingRegressor(n_estimators=100)
    gb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    test_predictions = gb_model.predict(X_test.reshape(X_test.shape[0], -1))
    future_X = X_test[-1].reshape(1, -1, 1)
    future_predictions = []
    for _ in range(days):
        pred = gb_model.predict(future_X.reshape(1, -1))
        future_predictions.append(pred[0])
        future_X = np.roll(future_X, -1)
        future_X[0, -1, 0] = pred
elif model_choice == "XGBoost":
    xgb_model = xgb.XGBRegressor(n_estimators=100)
    xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    test_predictions = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    future_X = X_test[-1].reshape(1, -1, 1)
    future_predictions = []
    for _ in range(days):
        pred = xgb_model.predict(future_X.reshape(1, -1))
        future_predictions.append(pred[0])
        future_X = np.roll(future_X, -1)
        future_X[0, -1, 0] = pred

# Reverse scaling for predictions
test_predictions = price_scaler.inverse_transform(test_predictions.reshape(-1, 1))
future_predictions = price_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
y_test = price_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mse = mean_squared_error(y_test, test_predictions)
tmse = np.sum((y_test - test_predictions) ** 2)
r2 = r2_score(y_test, test_predictions)

# Create future dates for the prediction plot
future_dates = pd.date_range(start=crypto_data.index[-1], periods=days + 1, freq='D')[1:]

# Combine historical, test, and future predictions
all_predictions = np.concatenate([test_predictions, future_predictions])
all_dates = np.concatenate([crypto_data.index[-(len(y_test)):], future_dates])

# Plot results with Plotly
fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=crypto_data.index,
    y=price_scaler.inverse_transform(crypto_data['price'].values.reshape(-1, 1)).flatten(),
    mode='lines',
    name='Historical Price'
))

# Prediction data
fig.add_trace(go.Scatter(
    x=all_dates,
    y=all_predictions.flatten(),
    mode='lines',
    name='Prediction',
    line=dict(color='red')
))

# Update layout for better visuals
fig.update_layout(
    title=f'{crypto.capitalize()} Price Prediction ({model_choice})',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified'
)

# Show plot in Streamlit
st.plotly_chart(fig)

# Display metrics
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Total Mean Squared Error (TMSE): {tmse}")
st.write(f"R² Score: {r2}")

# Calculate and display TWAP and VWAP
twap = calculate_twap(price_scaler.inverse_transform(crypto_data['price'].values.reshape(-1, 1)).flatten())
vwap = calculate_vwap(price_scaler.inverse_transform(crypto_data['price'].values.reshape(-1, 1)).flatten(), volume_scaler.inverse_transform(crypto_data['volume'].values.reshape(-1, 1)).flatten())

st.write(f"Time-Weighted Average Price (TWAP): {twap}")
st.write(f"Volume-Weighted Average Price (VWAP): {vwap}")
