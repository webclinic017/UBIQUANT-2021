'''
v_1.4
alpha101单因子测试
'''

from __future__ import print_function
import logging


import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.optimize as sco
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed



import warnings
warnings.filterwarnings("ignore")



############################ Define our factors ####################################### 
def RSIIndividual(data):
    t = 6                           # 需要6天的历史数据
    close = np.array(data)[:, 5]    # 股票N天的close数据
    delta = np.diff(close)     

    delta = pd.Series(delta)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=5).mean()                      # 上涨天数的指数移动平均
    roll_down1 = down.abs().ewm(span=5).mean()            # 下跌天数的指数移动平均
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    return -RSI1.iloc[-1]


def SixDayRS(data):
    '''
    六日收益率反转
    '''
    t = 6                           # 需要6天的历史数据
    close = np.array(data)[:, 5]    # 股票N天的close数据    
    return -(close[-1]/close[-t] - 1) 

def MOM(data):
    t = 6                           # 需要6天的历史数据
    close = np.array(data)[:, 5]    # 股票N天的close数据
    delta = np.diff(close)     

    delta = pd.Series(delta)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=5).mean()                      # 上涨天数的指数移动平均
    roll_down1 = down.abs().ewm(span=5).mean()            # 下跌天数的指数移动平均
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    six_day_rs = -(close[-1]/close[-t] - 1) 

    return (-RSI1.iloc[-1] + six_day_rs)/2

def Size(data):
    '''
    市值（收盘价近似代替）
    '''
    t = 6                                   # 需要6天的历史数据
    close = np.array(data)[:, 5]            # 股票N天的close数据
    size = np.log(close)         # 按行求均值
    size = pd.Series(size)
    size = size.rolling(window=5).mean()

    return -size.iloc[-1]


def VolumePct(data):
    '''
    成交量变化率
    '''
    t = 6                                      # 需要6天的历史数据
    volume = np.array(data)[:, 6]              # 股票N天的volume数据
    
    volume = pd.Series(volume)
    # v_pct = np.abs(volume.pct_change(1))
    v_pct = volume.pct_change(3)
    v_pct = v_pct.rolling(window=5).mean()
    return -v_pct.iloc[-1]

###############################   Alpha 101   ##############################
def Alpha6(data):
    _open = pd.Series(np.array(data)[:, 2])                # 开盘价
    volume = pd.Series(np.array(data)[:, 6])              # 股票N天的volume数据
    alpha6 = _open.rolling(window=10).corr(volume)
    return -1*alpha6.iloc[-1]

def Alpha7(data):
    close = pd.Series(np.array(data)[:, 5])
    amount = pd.Series(np.array(data)[:, 7])              # 股票N天的amount数据
    mean_amount = amount.rolling(window=5).mean().iloc[-1]
    if mean_amount<amount.iloc[-1]:
        delta = close.diff(3).abs().iloc[-10:]
        delta_rank = delta.rank()
        return -1*delta_rank.iloc[-1]*np.sign(delta.iloc[-1])
    else:
        return -1

def Alpha12(data):
    close = pd.Series(np.array(data)[:, 5])
    amount = pd.Series(np.array(data)[:, 7])              # 股票N天的amount数据
    _sign = np.sign(amount.diff(1).iloc[-1])
    delta = close.diff(1)
    return -1*_sign*delta.iloc[-1]

def Alpha23(data):
    high = pd.Series(np.array(data)[:, 3])
    _mean = high.rolling(10).mean()
    _delta = high.diff(2)
    if _mean.iloc[-1]<high.iloc[-1]:
        return -1*_delta.iloc[-1]
    else:
        return 0

def Alpha26(data):
    high = pd.Series(np.array(data)[:, 3])[-10:]
    amount = pd.Series(np.array(data)[:, 7])[-10:]
    high_rank = high.rank()
    amount_rank = amount.rank()
    _corr = amount_rank.rolling(window=5).corr(high_rank)
    res = _corr.rolling(window=3).max()

    return -1*res.iloc[-1]

def Alpha28(data):
    high = pd.Series(np.array(data)[:, 3])
    low = pd.Series(np.array(data)[:, 4])
    close = pd.Series(np.array(data)[:, 5])
    amount = pd.Series(np.array(data)[:, 7])[-10:]
    adv10 = amount.rolling(window=10).mean().fillna(0)
    _corr = adv10.rolling(window=5).corr(low).fillna(0)
    _scale = _corr+(high+low)/2-close
    _scale /= _scale.abs().sum()
    
    return _scale.iloc[-1]

###############################   并行计算因子   ##############################
def each_task(_key, _array):
    factors = []
    # factors.append(RSIIndividual(_array))
    # factors.append(SixDayRS(_array))
    # factors.append(Size(_array))
    # factors.append(VolumePct(_array))
    factors.append(Alpha28(_array))
    return (_key, factors)

def mp_task(batch, data):
    res = []
    for _key in batch:
        res.append(each_task(_key, data[_key]))
    return res

def _batch(stock_list):
    num_of_stock = len(stock_list)
    batch_size = num_of_stock // os.cpu_count()
    for i in range(0, num_of_stock, batch_size):
        yield stock_list[i: i+batch_size]

###############################   回归法估计因子权重   ##############################
def factor_regression(factors_record, pct_record, i):
    X = factors_record[-(i+2)]
    X[np.isnan(X)] = 0
    Y = pct_record[-i]
    Y[np.isnan(Y)] = 0
    reg = LinearRegression()
    # print("X", X)
    # print("Y", Y)
    reg.fit(X,Y)   
    factor_weight = list(reg.coef_)
    factor_weight.append(reg.intercept_)
    
    return factor_weight

###############################   有效前沿计算个股权重   ##############################
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

###############################   计算仓位收益率   ##############################
def calculate_pct(all_data, position, my_sequence):

    # t+2 和 t+1 的价差
    delta = all_data[all_data['Date'] == my_sequence+2]['Close'].values - \
            all_data[all_data['Date'] == my_sequence+1]['Close'].values

    profit = position @ delta
    return profit



# 因子或是回归最多利用过去10天的数据
MAX_NUM_RECORD = 10
num_of_holding = 100
leverage = 1
my_sequence = 0        # 当前日期
cur_capital = 5e8          # 初始资金
fee = 1.5e-4           # 手续费率

data = {}               # 储存一段时间的行情
yesterday_close = []    # 记录昨日收盘价，用于计算当日收益率
pct_record = []         # 记录每日收益率
factors_record = []     # 记录每日因子值
factors_weight = []     # 因子加权权重
capital_record = []     # 记录资金变动

file_name = r"./CONTEST_DATA_TEST_100_1.csv"
all_data = pd.read_csv(file_name, 
                    usecols=[0,1,2,3,4,5,6,7], 
                    names=['Date','Instrument','Open','High','Low','Close','Volume','Amount'])


daily_data = all_data[all_data['Date']==my_sequence].values
num_of_stock = daily_data.shape[0]
stock_list = np.arange(num_of_stock)

print(daily_data.shape)

# 每一支股票单独存在字典里
for i in range(num_of_stock):
    data[i] = []
    data[i].append(daily_data[i])

stock_batch = list(_batch(stock_list))
# print(stock_batch)

total_days = 900
mp.set_start_method('fork')

while(my_sequence<total_days-5):
    my_sequence += 1
    all_factors = [[]] * num_of_stock     # 每个股票的所有因子是一行
    
    start =  time.time()
    capital_record.append(cur_capital-5e8)
    print('current day: {}, current capital: {}......\n'.format(my_sequence, cur_capital))

    daily_data = all_data[all_data['Date']==my_sequence].values
    daily_data = daily_data[:, 0:8]
    today_close = daily_data[:, 5]                    # 当日收盘价
    if len(yesterday_close)>0:                        # 从第二天开始算起
        pct = (today_close - yesterday_close)/yesterday_close
        pct_record.append(pct)

    num_of_stock = daily_data.shape[0]
    stock_list = np.arange(num_of_stock)

    print(daily_data.shape)

    # 每一支股票单独存在字典里
    for i in range(num_of_stock):
        data[i].append(daily_data[i])

    use_time = time.time() - start
    print("Time of 🍦: ", round(use_time,2))

    if len(data[0]) >= 7:

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(mp_task, batch, data) for batch in stock_batch]
            for future in as_completed(futures):
                factors_batch = future.result()
                for i in range(len(factors_batch)):
                    _key, factors = factors_batch[i]
                    # print(_key, factors)
                    all_factors[_key] = factors

        # print(all_factors)
        print(len(all_factors))                                               # N * K

        use_time = time.time() - start
        print("Time of 🍎: ", round(use_time,2))

        all_factors = np.array(all_factors)
        all_factors[np.isnan(all_factors)] = 0                                # 处理空值
        all_factors[np.isinf(all_factors)] = 0                                # 处理无限值
        all_factors = preprocessing.scale(all_factors)              # 按列z-score标准化
        # all_factors = np.hstack(([[1]]*num_of_stock, all_factors))            # 每一行最左侧插入[1]
        factors_record.append(all_factors)                                    # T * N * (K)，储存一段时间的因子值，用于计算因子收益率
        num_of_factor = all_factors.shape[1]
        
        if len(factors_weight) == 0:
            final_factors = np.array(all_factors).mean(axis=1)
        else:
            final_factors = np.hstack((all_factors, [[1]]*num_of_stock)) @ np.array(factors_weight)
        
        use_time = time.time() - start
        print("Time of 🍌: ", round(use_time,2))

        _rank = np.argsort(np.argsort(final_factors))    # 0-499，多因子等权组合之后的排序值
        _pos = np.array([0]*num_of_stock)
        
        _buy_line = num_of_stock-num_of_holding/2
        _sell_line = num_of_holding/2
        _pos[_rank>=_buy_line] = 1
        _pos[_rank<_sell_line] = -1
        
        position = _pos * cur_capital * leverage / num_of_holding / today_close
        position[np.isnan(position)] = 0
        position = position.astype(int)
        print("len of position: ", len(position))
        # print(position)
        daily_profit = calculate_pct(all_data, position, my_sequence)
        cur_capital += daily_profit
        cur_capital *= (1-fee)          # 扣除手续费

        use_time = time.time() - start
        print("Time of 🍊: ", round(use_time,2))

        if len(factors_record) >= 10:
            # 其实只做了8次回归，取平均值
            factors_weight = []
            
            # with ProcessPoolExecutor() as executor:
            #     futures = [executor.submit(factor_regression, factors_record, pct_record, i+1) for i in range(len(factors_record)-2)]
            #     for future in as_completed(futures):
            #         factors_weight.append(future.result())
            
            # factors_weight = np.array(factors_weight).mean(axis=0)
            # factors_weight /= np.sum(factors_weight)

            # print("factors_weight: ", factors_weight)
            # if len(factors_record) >= 20: factors_record.pop(0)

        use_time = time.time() - start
        print("Time of 🍐: ", round(use_time,2))

        # 删除最老的一天的数据
        if len(data[0]) >= 20:
            for i in range(num_of_stock):
                data[i].pop(0)
        
        use_time = time.time() - start
        print("Time of 🍉: ", round(use_time,2))

    # use_time = time.time() - start
    # print("Time of posting position: %s", use_time)
    # time.sleep(0.1)
    
    yesterday_close = today_close

total_return = round(capital_record[-1]/5e8, 4)
daily_pct = pd.Series(capital_record)+5e8
daily_pct = daily_pct.pct_change()
sharpe = daily_pct.mean() / daily_pct.std() * np.sqrt(252)
sharpe = np.round(sharpe)

_name = 'alpha28'
plt.figure()
plt.plot(capital_record)
_title = "{}, ret:{}, sharpe:{}".format(_name, total_return, sharpe)
plt.title(_title)
plt.savefig('backtest_{}.png'.format(_name))

factors_record = np.array(factors_record)[:,:,0]
pd.DataFrame(factors_record).to_csv('factors_{}.csv'.format(_name))