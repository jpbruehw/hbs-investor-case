# ========================== Imports ===================================
import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import cvxpy as cp

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

start_date = '1997-01-01'
end_date = '2016-12-31'
get_index_data('^STI', start_date, end_date, 'index-data-sti.xlsx')
get_index_data('^GSPC', start_date, end_date, 'index-data-sp500.xlsx')

# ==================== Step 2: Calculate CAGR ==========================

cagr, cumulative_return = calculate_cagr(
    start_value=100000,
    future_value=500000,
    annual_contribution=10000,
    n_years=20
)
print(f"CAGR: {cagr:.4f}, Cumulative Return: {cumulative_return:.4f}")

# ==================== Step 3: Required Returns =========================

net_flows = [10000, -20000, -50000, -50000, -20000]
returns, avg_required_return = required_returns(500000, net_flows)
print(f"Average Required Return: {avg_required_return:.4f}")

# =================== Step 4: Optimal Portfolio Construction ============

# Load ETF data
df_base = pd.read_excel('./raw-data/etfs-reit-gold.xlsx')
start_date = str(df_base['Month'][0]).split()[0]
end_date = str(df_base['Month'].iloc[-1]).split()[0]

# Fetch asset close prices
tickers = ['VNQ', 'GLD']
asset_data = fetch_asset_data(tickers, start_date, end_date)

# Combine data
df_base.drop(columns=['Month'], inplace=True)
asset_data.set_index(df_base.index, inplace=True)
full_df = pd.concat([df_base, asset_data], axis=1)

# Calculate returns and stats
returns_df = full_df.pct_change().dropna()
cov_matrix = annualized_covariance(returns_df)
std_avg = mean_std(returns_df)
expected_returns = annualized_return(returns_df)
n_assets = len(expected_returns)

# =================== Step 5: Portfolio Optimization ===================

weights = cp.Variable(n_assets)
constraints = [weights >= 0, cp.sum(weights) == 1]
target_std = 0.0959
risk_free_rate = 0.0087

portfolio_risk = cp.quad_form(weights, cov_matrix) ** 0.5
portfolio_return = expected_returns.values @ weights

problem = cp.Problem(cp.Maximize(portfolio_return), constraints + [portfolio_risk <= target_std])
problem.solve(solver=cp.ECOS, qcp=True, verbose=True)

print("Optimized Weights:", weights.value)
print("Expected Return:", portfolio_return.value)
print("Standard Deviation:", portfolio_risk.value)
print("Sharpe Ratio:", (portfolio_return.value - risk_free_rate) / portfolio_risk.value)

# ================== Step 6: Efficient Frontier =======================

target_returns = np.linspace(0, max(expected_returns), 10000)
portfolio_std_dev, portfolio_returns = [], []

for target_r in target_returns:
    prob = cp.Problem(cp.Minimize(cp.quad_form(weights, cov_matrix)),
                    constraints + [weights.T @ expected_returns.values == target_r])
    prob.solve()
    portfolio_std_dev.append(np.sqrt(prob.value))
    portfolio_returns.append(target_r)

portfolio_std_dev = np.array(portfolio_std_dev)
portfolio_returns = np.array(portfolio_returns)
sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_std_dev

# Min variance portfolio
min_std_idx = np.argmin(portfolio_std_dev)
print("Min Std Dev:", portfolio_std_dev[min_std_idx])
print("Return at Min Std:", portfolio_returns[min_std_idx])
print("Sharpe at Min Std:", sharpe_ratios[min_std_idx])

# Export to Excel
pd.DataFrame({
    'portfolio_std_dev': portfolio_std_dev,
    'portfolio_return': portfolio_returns,
    'sharpe_ratio': sharpe_ratios
}).to_excel('./results/mean-var-data.xlsx', index=False)
