# ========================== Imports ===================================
import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import cvxpy as cp

# ========================== Global Vars ===============================

# Index settings
INDEX_TICKERS = {
    'STI': '^STI',
    'SP500': '^GSPC',
}
INDEX_OUTPUT_FILES = {
    'STI': 'index-data-sti.xlsx',
    'SP500': 'index-data-sp500.xlsx',
}
INDEX_START_DATE = '1997-01-01'
INDEX_END_DATE = '2016-12-31'

# CAGR settings
# Underscore (_) as 0000 separator for readability
CAGR_START = 100_000
CAGR_FUTURE = 500_000
CAGR_ANNUAL_CONTRIB = 10_000
CAGR_YEARS = 20
CAGR_DELAY = 1

# Required returns
NET_FLOWS = [10_000, -20_000, -50_000, -50_000, -20_000]
TARGET_VALUE = 500_000

# Asset optimization
ETF_DATA_PATH = './raw-data/etfs-reit-gold.xlsx'
ADDITIONAL_TICKERS = ['VNQ', 'GLD']
TARGET_STD = 0.0959
RISK_FREE_RATE = 0.0087
RESULTS_PATH = './results/mean-var-data.xlsx'
NUM_POINTS_EFF_FRONTIER = 10_000

# ========================== Helper Functions ==========================

def get_index_data(ticker, start_date, end_date, filename):
    data = yf.Ticker(ticker).history(period='1d', start=start_date, end=end_date)
    df = pd.DataFrame({'date': data.index.strftime('%Y-%m-%d'), ticker: data['Close'].values})
    df.to_excel(filename, index=False)
    return df

def calculate_cagr(start_value, future_value, annual_contribution, n_years, delay=1):
    rate = npf.rate(n_years - delay, -annual_contribution, -start_value, future_value)
    cumulative_return = ((1 + rate) ** n_years) - 1
    return rate, cumulative_return

def required_returns(target_value, cash_flows):
    returns = [((target_value - cf) / target_value) - 1 for cf in cash_flows]
    return returns, np.mean(returns)

def fetch_asset_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    for ticker in tickers:
        close = yf.Ticker(ticker).history(start=start_date, end=end_date)['Close']
        data[ticker] = close.resample('ME').last()
    return data

def annualized_return(df):
    return ((1 + df).prod()) ** (12 / len(df)) - 1

def annualized_covariance(df):
    return df.cov() * 12

def mean_std(df):
    return df.std().mean() * np.sqrt(12)

# ==================== Step 1: Export Index Data =======================

for name, ticker in INDEX_TICKERS.items():
    get_index_data(ticker, INDEX_START_DATE, INDEX_END_DATE, INDEX_OUTPUT_FILES[name])

# ==================== Step 2: Calculate CAGR ==========================

cagr, cumulative_return = calculate_cagr(
    start_value=CAGR_START,
    future_value=CAGR_FUTURE,
    annual_contribution=CAGR_ANNUAL_CONTRIB,
    n_years=CAGR_YEARS,
    delay=CAGR_DELAY
)
print(f"CAGR: {cagr:.4f}, Cumulative Return: {cumulative_return:.4f}")

# ==================== Step 3: Required Returns =========================

returns, avg_required_return = required_returns(TARGET_VALUE, NET_FLOWS)
print(f"Average Required Return: {avg_required_return:.4f}")

# =================== Step 4: Optimal Portfolio Construction ============

df_base = pd.read_excel(ETF_DATA_PATH)
start_date = str(df_base['Month'][0]).split()[0]
end_date = str(df_base['Month'].iloc[-1]).split()[0]

asset_data = fetch_asset_data(ADDITIONAL_TICKERS, start_date, end_date)

df_base.drop(columns=['Month'], inplace=True)
asset_data.set_index(df_base.index, inplace=True)
full_df = pd.concat([df_base, asset_data], axis=1)

returns_df = full_df.pct_change().dropna()
cov_matrix = annualized_covariance(returns_df)
std_avg = mean_std(returns_df)
expected_returns = annualized_return(returns_df)
n_assets = len(expected_returns)

# =================== Step 5: Portfolio Optimization ===================

weights = cp.Variable(n_assets)
constraints = [weights >= 0, cp.sum(weights) == 1]

portfolio_risk = cp.quad_form(weights, cov_matrix) ** 0.5
portfolio_return = expected_returns.values @ weights

problem = cp.Problem(cp.Maximize(portfolio_return), constraints + [portfolio_risk <= TARGET_STD])
problem.solve(solver=cp.ECOS, qcp=True, verbose=True)

print("Optimized Weights:", weights.value)
print("Expected Return:", portfolio_return.value)
print("Standard Deviation:", portfolio_risk.value)
print("Sharpe Ratio:", (portfolio_return.value - RISK_FREE_RATE) / portfolio_risk.value)

# ================== Step 6: Efficient Frontier =======================

target_returns = np.linspace(0, max(expected_returns), NUM_POINTS_EFF_FRONTIER)
portfolio_std_dev, portfolio_returns = [], []

for target_r in target_returns:
    prob = cp.Problem(cp.Minimize(cp.quad_form(weights, cov_matrix)),
                      constraints + [weights.T @ expected_returns.values == target_r])
    prob.solve()
    portfolio_std_dev.append(np.sqrt(prob.value))
    portfolio_returns.append(target_r)

portfolio_std_dev = np.array(portfolio_std_dev)
portfolio_returns = np.array(portfolio_returns)
sharpe_ratios = (portfolio_returns - RISK_FREE_RATE) / portfolio_std_dev

min_std_idx = np.argmin(portfolio_std_dev)
print("Min Std Dev:", portfolio_std_dev[min_std_idx])
print("Return at Min Std:", portfolio_returns[min_std_idx])
print("Sharpe at Min Std:", sharpe_ratios[min_std_idx])

pd.DataFrame({
    'portfolio_std_dev': portfolio_std_dev,
    'portfolio_return': portfolio_returns,
    'sharpe_ratio': sharpe_ratios
}).to_excel(RESULTS_PATH, index=False)