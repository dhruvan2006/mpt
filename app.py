import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_asset_data(tickers):
    "Fetch the `close` prices of an array of tickers and returns a DataFrame"
    data = {}
    for ticker in tickers:
        df = yf.download(ticker)
        if not df.empty:
            data[ticker] = df['Close']
    return pd.DataFrame(data).dropna()

ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']
DEFAULT_ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
MAX_PORTFOLIOS = 10000

selected_assets = st.multiselect("Select your assets:", ASSETS, default=DEFAULT_ASSETS)

num_portfolios = st.slider('Number of Portfolios', min_value=0, max_value=MAX_PORTFOLIOS, value=1000, step=1000)

df = get_asset_data(selected_assets)

returns = df.pct_change().dropna()
mean = returns.mean()
cov_matrix = returns.cov()

weights = np.random.random((num_portfolios, len(selected_assets )))
weights /= np.sum(weights, axis=1)[:, np.newaxis]

port_returns = np.dot(weights, mean) * 365
port_risk = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix * 365, weights))

sharpe_ratios = port_returns / port_risk

max_sharpe_idx = np.argmax(sharpe_ratios)

hover_text = [f"{', '.join([f'{asset}: {weight:.2%}' for asset, weight in zip(selected_assets, w)])}<br>Sharpe Ratio: {sharpe:.2f}" for w, sharpe in zip(weights, sharpe_ratios)]
tangency_weights = weights[max_sharpe_idx]
tangency_annotation_text = f"Tangency Portfolio<br>Weights: {', '.join([f'{asset}: {weight:.2%}' for asset, weight in zip(selected_assets, tangency_weights)])}"

fig = go.Figure(
    go.Scatter(
        x=port_risk,
        y=port_returns,
        mode='markers',
        marker=dict(
            color=sharpe_ratios,
            colorscale='viridis',
            colorbar=dict(
                title='Sharpe Ratio'
            )
        ),
        text=hover_text,
        hoverinfo='text',
    )
)

for asset in selected_assets:
    asset_returns = returns[asset]
    asset_std = asset_returns.std() * np.sqrt(365)
    asset_mean = asset_returns.mean() * 365
    fig.add_trace(go.Scatter(x=[asset_std], y=[asset_mean], mode='markers', marker=dict(color='red', size=10), name=asset))
    fig.add_annotation(x=asset_std, y=asset_mean+0.05, text=asset, showarrow=False, font=dict(color="white", size=12))

fig.add_trace(go.Scatter(x=[port_risk[max_sharpe_idx]], y=[port_returns[max_sharpe_idx]], mode='markers', marker=dict(color='blue', size=10), name='Tangency Portfolio'))
fig.add_annotation(x=port_risk[max_sharpe_idx], y=port_returns[max_sharpe_idx]+0.05, text=tangency_annotation_text, showarrow=False, font=dict(color="white", size=12))

st.plotly_chart(fig)

st.text('Data')
st.dataframe(df)
