import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Function to fetch asset data
def get_asset_data(tickers):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker)
        if not df.empty:
            data[ticker] = df['Close']
    return pd.DataFrame(data).dropna()

# Function to calculate the efficient frontier
def calculate_efficient_frontier(returns, num_points=100):
    mean = returns.mean()
    cov_matrix = returns.cov()
    
    def portfolio_risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
    
    def portfolio_return(weights):
        return np.dot(mean, weights) * 365

    def get_efficient_portfolio(target_return):
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: portfolio_return(weights) - target_return}
        )
        bounds = tuple((0, 1) for _ in range(len(mean)))
        result = minimize(portfolio_risk, len(mean) * [1. / len(mean)], bounds=bounds, constraints=constraints)
        return result.fun
    
    target_returns = np.linspace(returns.mean().min() * 365, returns.mean().max() * 365, num_points)
    risks = [get_efficient_portfolio(tr) for tr in target_returns]
    
    return risks, target_returns

ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']
DEFAULT_ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
MAX_PORTFOLIOS = 10000

selected_assets = st.multiselect("Select your assets:", ASSETS, default=DEFAULT_ASSETS)
num_portfolios = st.slider('Number of Portfolios', min_value=0, max_value=MAX_PORTFOLIOS, value=5000, step=1000)
risk_free_rate = st.number_input('Risk free Rate (in %)', value=0.0, step=0.1) / 100

df = get_asset_data(selected_assets)

returns = df.pct_change().dropna()
mean = returns.mean()
cov_matrix = returns.cov()

weights = np.random.random((num_portfolios, len(selected_assets )))
weights /= np.sum(weights, axis=1)[:, np.newaxis]

port_returns = np.dot(weights, mean) * 365
port_risk = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix * 365, weights))

sharpe_ratios = (port_returns - risk_free_rate) / port_risk

max_sharpe_idx = np.argmax(sharpe_ratios)

hover_text = [f"{', '.join([f'{asset}: {weight:.2%}' for asset, weight in zip(selected_assets, w)])}<br>Sharpe Ratio: {sharpe:.2f}" for w, sharpe in zip(weights, sharpe_ratios)]
tangency_weights = weights[max_sharpe_idx]
tangency_annotation_text = f"Tangency Portfolio<br>{', '.join([f'{asset}: {weight:.2%}' for asset, weight in zip(selected_assets, tangency_weights)])}"

risks, target_returns = calculate_efficient_frontier(returns)

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

fig.add_trace(go.Scatter(
    x=risks, y=target_returns, 
    mode='lines', 
    line=dict(color='blue', width=2),
    name='Efficient Frontier'
))

for asset in selected_assets:
    asset_returns = returns[asset]
    asset_std = asset_returns.std() * np.sqrt(365)
    asset_mean = asset_returns.mean() * 365
    fig.add_trace(go.Scatter(x=[asset_std], y=[asset_mean], mode='markers', marker=dict(color='red', size=10), name=asset))
    fig.add_annotation(x=asset_std, y=asset_mean+0.05, text=asset, showarrow=False, font=dict(color="white", size=12))

fig.add_trace(go.Scatter(x=[port_risk[max_sharpe_idx]], y=[port_returns[max_sharpe_idx]], mode='markers', marker=dict(color='blue', size=10), name='Tangency Portfolio'))
fig.add_annotation(x=port_risk[max_sharpe_idx], y=port_returns[max_sharpe_idx]+0.05, text=tangency_annotation_text, showarrow=False, font=dict(color="red", size=12))

fig.update_layout(
    margin=dict(l=40, r=40, t=40, b=40),
    height=600,
    width=800,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig)

st.subheader('Selected Asset Data')
st.dataframe(df)

st.subheader('Portfolio Performance')
st.write(f"Maximum Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.2f}")
st.write(f"Tangency Portfolio Weights: {', '.join([f'{asset}: {weight:.2%}' for asset, weight in zip(selected_assets, tangency_weights)])}")
