import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# Default tickers
default_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']

# Additional tickers to choose from
additional_tickers = [
    'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'SHIB-USD', 'AVAX-USD',
    '^SPX', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
    'TLT'
]

# Function to fetch asset data
def get_ticker_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker)
            data[ticker] = df['Close']
        except Exception as e:
            st.error(f"Error fetching data for ticker {ticker}: {e}")
    return pd.DataFrame(data).dropna()

st.set_page_config(page_title="Modern Portfolio Theory", page_icon="🦈", layout="wide")

st.title("Modern Portfolio Theory")

# Streamlit UI for asset selection
selected_tickers = st.multiselect('Select tickers:', options=default_tickers+additional_tickers, default=default_tickers)
custom_tickers = st.text_input('Add custom tickers (comma separated):', '', placeholder='Stocks will skew the result because they are not traded 24/7')

if custom_tickers:
    custom_tickers_list = [ticker.strip() for ticker in custom_tickers.split(',')]
    selected_tickers += custom_tickers_list

# Streamlit UI for selecting risk free rate and number of portfolios
col1, col2 = st.columns(2)
with col1:
    risk_free_rate = st.number_input('Risk free rate (in %):', 0., 100., 0., 0.1)
with col2:
    num_portfolios = st.number_input('Number of portfolios to simulate', 1, 10000, 5000, 1000)

# Helper functions for efficient frontier
def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = np.sum(mean_returns * weights) * 365
    port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 365, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_stddev
    return np.asarray([port_return, port_stddev, sharpe_ratio])

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    return -calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# Main logic of the app
if selected_tickers:
    # Request data
    df = get_ticker_data(selected_tickers)

    # Max sharpe ratio
    returns = df.pct_change().dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(selected_tickers)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets])    # Just make equal weights for it to optimize
    optimal_weights = minimize(negative_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, risk_free_rate), bounds=bounds, constraints=constraints)

    max_sharpe_ratio_return, max_sharpe_ratio_volatility, max_sharpe_ratio = calculate_portfolio_performance(optimal_weights.x, mean_returns, cov_matrix, risk_free_rate)

    # Efficient frontier
    target_returns = np.linspace(mean_returns.min() * 365, mean_returns.max() * 365, 100)
    efficient_portfolios = []

    for target_return in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix, risk_free_rate)[0] - target_return})
        optimal_weights = minimize(negative_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, risk_free_rate), bounds=bounds, constraints=constraints)
        portfolio_return, portfolio_volatility, _ = calculate_portfolio_performance(optimal_weights.x, mean_returns, cov_matrix, risk_free_rate)
        efficient_portfolios.append((optimal_weights.x, portfolio_return, portfolio_volatility))
    
    # Plot randomly generated portfolios
    random_weights = np.random.random((num_portfolios, num_assets)) ** 2  # Bias it slightly towards the extremes
    random_weights /= np.sum(random_weights, axis=1)[:, np.newaxis]

    portfolio_returns_random = np.dot(random_weights, mean_returns) * 365
    portfolio_stddevs_random = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix * 365, random_weights))

    sharpe_ratios_random = (portfolio_returns_random - risk_free_rate) / portfolio_stddevs_random

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_stddevs_random,
        y=portfolio_returns_random,
        mode='markers',
        marker=dict(color=sharpe_ratios_random, colorscale='viridis', colorbar=dict(title='Sharpe Ratio')),
        name='Randomly generated portfolios'
    ))

    # Plot individual assets
    annual_volatility = returns.std() * np.sqrt(365)
    sharpe_ratios = (mean_returns * 365 - risk_free_rate) / annual_volatility

    text_labels = [f"{ticker}: {sharpe:.2f}" for ticker, sharpe in zip(selected_tickers, sharpe_ratios)]


    fig.add_trace(go.Scatter(
        x=annual_volatility,
        y=mean_returns * 365,
        mode='markers+text',
        marker=dict(color='red', size=8),
        text=text_labels,
        textposition='top center',
        name='Tickers'
    ))
    fig.update_layout(
        title='Modern Portfolio Theory',
        xaxis_title='Standard deviation (of annual returns)',
        yaxis_title='Expected annual return',
        xaxis=dict(showgrid=True, tickformat=".2%"),
        yaxis=dict(showgrid=True, tickformat=".2%")
    )

    # Plot max sharpe ratio
    efficient_returns = [p[1] for p in efficient_portfolios]
    efficient_volatilities = [p[2] for p in efficient_portfolios]

    fig.add_trace(go.Scatter(
        x=[max_sharpe_ratio_volatility],
        y=[max_sharpe_ratio_return],
        mode='markers+text',
        marker=dict(color='blue', size=10),
        text=f'Max Sharpe Ratio {max_sharpe_ratio:.2f}',
        textposition='top left',
        name='Tangency Portfolio'
    ))

    # Plot efficient frontier
    fig.add_trace(go.Scatter(
        x=efficient_volatilities,
        y=efficient_returns,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='white', width=2)
    ))

    # Plot tangent CAL
    min_volatility = min(efficient_volatilities)
    max_volatility = max(efficient_volatilities)
    tangent_volatility = [min_volatility, max_volatility]
    tangent_return = [max_sharpe_ratio * volatility + risk_free_rate for volatility in tangent_volatility]

    fig.add_trace(go.Scatter(
        x=tangent_volatility,
        y=tangent_return,
        mode='lines',
        name='Tangent Line',
        line=dict(color='purple', width=2, dash='dash')
    ))

    fig.update_layout(
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    st.plotly_chart(fig)


    # Plot historical performance
    fig = make_subplots(rows=len(selected_tickers), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, ticker in enumerate(selected_tickers):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[ticker], mode='lines', name=ticker),
            row=i+1, col=1
        )
    fig.update_yaxes(type="log")
    fig.update_layout(height=200 * len(selected_tickers), title_text="Historical Performance")
    st.plotly_chart(fig)
else:
    st.warning("Please select at least one ticker.")

