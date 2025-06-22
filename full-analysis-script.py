# import packages
import pandas as pd
import numpy as np, numpy.random
import yfinance as yf

# get data for sti
ticker_sti = '^STI'
start_date = '1997-01-01'
end_date = '2016-12-31'
sti_data = yf.Ticker(ticker_sti).history(period='1d', start=start_date, end=end_date)

# get data for sp500
ticker_sp500 = '^GSPC'
sp500_data = yf.Ticker(ticker_sp500).history(period='1d', start=start_date, end=end_date)

# create dataframe to excel file
df = pd.DataFrame()
df['sti'] = sti_data['Close']
df['date'] = df.index
df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

# export to excel
df.to_excel('index-data-sti.xlsx', index=False)

# create dataframe to excel file
df2 = pd.DataFrame()
df2['sp500'] = sp500_data['Close']
df2['date'] = df2.index
df2['date'] = df2['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

# export to excel
df2.to_excel('index-data-sp500.xlsx', index=False)

# import numpy_financial to calculate annual return
# goal is to find the annual compounded return assuming annual contributions
# frist step is to import packages
import numpy_financial as npf

# list known values
# we know the portfolio started at 100k
# twenty years later it is now 500k
# we can safely assume an annual contribution of 10k based on case study
# investing started twenty years ago with 
future_value = 500000
start_value = -100000
annual_contribution = 10000
n = 20

# we assume contributions started one year from t0
# therefore we subtract 1 from n
delayed_start = 1
adjusted_n = n - delayed_start

# calculate the CAGR using numpy-financial
cagr = npf.rate(adjusted_n, -annual_contribution, start_value, future_value)
print(f"The calculated CAGR is: {cagr:}")

# find cumulative return
cumulative_return = ((1 + cagr) ** n) - 1
print(cumulative_return)

# find required return to keep portfolio principal at 500k
#----------------------------------------------#

# we set the target val to the t0 val of portfolio 
target_val = 500000

# create an array of inflows and outflows
# inflow of 10k in t0, -30 + 10k inflow in t+1, etc.
net_flows = [10000, -20000, -50000, -50000, -20000]

# create list to store the required returns for each period
returns = []

# loop through each period
# calculate the required return to keep the portfolio stable
for cash_flow in net_flows:
    
    # calculate the required return
    # if the inflow is positive, the required return is negative and vice versa
    # calculates the ratio of the ending portfolio value to the target
    # this tells us how much the portfolio needs to increase/decrease to remain stable
    # the return is assuming we are able to maintain the target val over time by earning the required return
    required_return = ((target_val - cash_flow) / target_val) - 1

    # append the required return to the list
    returns.append(required_return)

# calculate the average return over the periods
sum_returns = sum(returns)
n_returns = len(returns)
avg_return = sum_returns / n_returns
print(avg_return)

# find optimal portfolio
#----------------------#

# import packages
import pandas as pd
import yfinance as yf

# import excel file of etfs
df = pd.read_excel('./raw-data/etfs-reit-gold.xlsx')

# get data from yfinance for the reit and gld
# extract the first and last values from the month column
# split the data and change to string to format
start_date = str(df['Month'][0]).split(' ')[0]
end_date = str(df['Month'].iloc[-1]).split(' ')[0]

# create dict to add values to
close_data = pd.DataFrame()

# create list of tickers for the reit and gld
tickers = ['VNQ', 'GLD']

# loop and call api to get data
for stock in tickers:
    
    # make call to yfinance
    data = yf.Ticker(stock).history(period='1d', start=start_date, end=end_date)
    
    # extract close
    # resample data to monthly
    # extract last day of month
    data = data['Close'].resample('M').last()
    
    # add to dictionary
    close_data[stock] = data
    
# drop month column
df.drop(columns=['Month'], inplace=True)

# line up the indices
close_data.set_index(df.index, inplace=True)

# add the dataframes
full_df = pd.concat([df, close_data], axis=1)

# calculate returns
returns_df = full_df.pct_change()[1:]

# get cov matrix
# annualized results
# annualized linearly
cov = returns_df.cov() * 12

# get standard deviation for each asset
std_per_asset = returns_df.std()

# get average
# annualized
average_std = std_per_asset.mean() * np.sqrt(12)
print(average_std)

# get return vector 
# annualized results
r = ((1 + returns_df).prod())**(12 / len(returns_df)) - 1

# define the number of assets
n_assets = len(r)

# import packages
import cvxpy as cp

# set weights
weights = cp.Variable(n_assets)

# constrain shortselling
# weights can't be zero and sum is 1
constraints = [weights >= 0, cp.sum(weights) == 1]

# set to 1 year t-bill rate for 2017
# about 1.7%
# convert from annual to monthly
risk_free_rate = 0.0087

# calculate and display portfolio weights

# create list to store results
optimized_portfolio = []

# define the target standard deviation
# average standard deviation is 14.47%
# assuming risk averse investor we can lower to 10% and see if solvable
# can't solve for 10% directly, so we lower slightly
target_std_dev = 0.0959

# add a constraint to limit the portfolio's standard deviation to the target value
portfolio_risk = cp.quad_form(weights, cov) ** 0.5
risk_constraint = portfolio_risk <= target_std_dev

# set up the objective function to maximize portfolio return
portfolio_return = r.values @ weights
objective = cp.Maximize(portfolio_return)

# create the optimization problem
problem = cp.Problem(objective, constraints + [risk_constraint])

# solve the optimization problem
problem.solve(qcp=True)

# store the results in dict
results = {
        'Target SD[r]': target_std_dev,
        'Portfolio Weights': weights.value,
        'Expected Return (E[r])': portfolio_return.value,
        'Standard Deviation (SD[r])': portfolio_risk.value
        }

# append results to list
optimized_portfolio.append(results['Portfolio Weights'])

# print results
print(portfolio_risk.value)
print(weights.value)
print(portfolio_return.value)

# get sharpe ratio of portfolio
sharpe_ratio = (portfolio_return.value - risk_free_rate) / portfolio_risk.value
print(sharpe_ratio)

# define a range of target returns
# create array of return possibilities
# max return is the highest return of individual stocks
# 1000 means 1000 increments (portfolios) between 0 and max
target_returns = np.linspace(0, max(r), 10000)

# initialize lists to store results
portfolio_std_dev = []
portfolio_return = []
sharpe_ratios = []
# loop through target returns to  find optimal portfolio
# finds the portfolio that tries to match the target return
# while minimizing the standard deviation
for target_return in target_returns:
    
    # set objective function to minimize portfolio variance
    # we then use this to find the standard deviation
    objective = cp.Minimize(cp.quad_form(weights, cov))
    
    # constraint to achieve the target return
    # transpose the weights to a row vector for multiplication
    constraint = [weights.T @ r == target_return]
    
    # solve the optimization problem
    prob = cp.Problem(objective, constraint + constraints)
    prob.solve()
    
    # calculate and store portfolio standard deviation and return
    portfolio_std_dev.append(np.sqrt(prob.value))
    portfolio_return.append(target_return)

# convert returns to np array to calculate the sharpe ratio for each return
portfolio_return = np.array(portfolio_return)

# calculate sharpe ratio
# risk free rate given in case
sharpe_ratios = (portfolio_return - 0.0087) / portfolio_std_dev

# convert std list to numpy arrays for plotting
portfolio_std_dev = np.array(portfolio_std_dev)

# find the minimum variance portfolio
# get index then callout datapoint
# we will use this data to make the efficient frontier plots
min_std_dev_index = np.argmin(portfolio_std_dev)
min_std_dev = portfolio_std_dev[min_std_dev_index]
mv_port_return = portfolio_return[min_std_dev_index]
sharpe_ratio_mean_var = sharpe_ratios[min_std_dev_index]
print(min_std_dev)
print(mv_port_return)
print(sharpe_ratio_mean_var)

# concat the two lists to a dataframe to create exportable csv
excel_export = pd.DataFrame()
excel_export['portfolio_std_dev'] = portfolio_std_dev
excel_export['portfolio_return'] = portfolio_return
excel_export['sharpe_ratio'] = portfolio_return

# export to excel
# use this data to calculate the weights for the optimal portfolio
# we can use a simple excel solver based on the return/std to find
excel_export.to_excel('./results/mean-var-data.xlsx', index=False)