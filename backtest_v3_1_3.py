'''
v_3_1_3

四因子 + Alpha101，去除相关性>0.3的，剩余8个因子
因子等权重组合
从第二天开始就可以算部分因子，加快开始交易的时间
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

import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


import warnings
warnings.filterwarnings("ignore")



############################ Define our factors ####################################### 

def SixDayRS(data):
    '''
    六日收益率反转
    '''
    t = 6                           # 需要6天的历史数据
    close = data[:, 5]              # 股票N天的close数据    
    # return (-(close[-1]/close[-t] - 1), _index)
    return -(close[-1]/close[-t] - 1)

###############################   Alpha 101   ##############################
def Alpha6(data):
    _open = pd.Series(data[:, 2])                  # 开盘价
    volume = pd.Series(data[:, 6])                 # 股票N天的volume数据
    alpha6 = _open.rolling(window=10).corr(volume) # 需要10天的历史数据
    return -1*alpha6.iloc[-1]


def Alpha9(data):
    close = pd.Series(data[:, 5])
    delta = close.diff(1)
    delta_min = delta.rolling(window=5).min().iloc[-1]
    if delta_min>0:
        return delta.iloc[-1]
    else:
        delta_max = delta.rolling(window=5).max().iloc[-1]
        if delta_max<0:
            return delta.iloc[-1]
        else:
            return -1*delta.iloc[-1]

def Alpha12(data):
    close = data[:, 5]
    amount = data[:, 7]                     # 股票N天的amount数据
    _sign = np.sign(amount[-1]-amount[-2])
    return -1*_sign*(close[-1]-close[-2])

def Alpha23(data):
    high = pd.Series(data[:, 3])
    _mean = high.rolling(10).mean()         # 需要10天数据
    _delta = high.diff(2)
    if _mean.iloc[-1]<high.iloc[-1]:
        return -1*_delta.iloc[-1]
    else:
        return 0

def Alpha51(data):
    close = data[:, 5]                      # 需要20天数据
    condition1 = (close[-20]-close[-10])/10 - (close[-10]-close[-1])/10
    if condition1 < -0.05:
        return 1
    else:
        return -1*(close[-1]-close[-2])

def Alpha53(data):
    high = data[:, 3]
    low = data[:, 4]
    close = data[:, 5]
    x1 = ((close[-1]-low[-1])-(high[-1]-close[-1]))/(close[-1]-low[-1])
    x9 = ((close[-9]-low[-9])-(high[-9]-close[-9]))/(close[-9]-low[-9])

    return -1*(x1-x9)

def Alpha54(data):
    _open = data[:, 2]
    _high = data[:, 3]
    _low = data[:, 4]
    _close = data[:, 5]
    fenzi = -1*(_low[-1]-_close[-1])*(_open[-1]**5)
    fenmu = (_low[-1]-_high[-1])*(_close[-1]**5)
    return fenzi/fenmu

###############################   并行计算因子   ##############################
def each_task(_key, _array):
    factors = []
    _array = np.array(_array)
    T = _array.shape[0]

    if T>=2:
        factors.append(Alpha12(_array))
        factors.append(Alpha54(_array))
    if T>=6:
        factors.append(Alpha9(_array))
        factors.append(SixDayRS(_array))
    if T>=10:
        factors.append(Alpha6(_array))
        factors.append(Alpha23(_array))
        factors.append(Alpha53(_array))
    if T>=20:
        factors.append(Alpha51(_array))
    
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



# 因子或是回归最多利用过去20天的数据
MAX_NUM_RECORD = 20
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

    if len(data[0]) >= 2:

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
        all_factors = preprocessing.scale(all_factors)                        # 按列z-score标准化
        # all_factors = Schimidt(all_factors)                                   # Schimidt正交化分解
        # all_factors = np.hstack(([[1]]*num_of_stock, all_factors))            # 每一行最左侧插入[1]
        factors_record.append(all_factors)                                    # T * N * (K)，储存一段时间的因子值，用于计算因子收益率
        num_of_factor = all_factors.shape[1]
        
        if len(factors_weight) == 0:
            final_factors = np.array(all_factors).mean(axis=1)                # 按行平均
        else:
            final_factors = all_factors @ factors_weight
        
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

        # 删除最老的一天的数据
        if len(data[0]) >= 20:
            for i in range(num_of_stock):
                data[i].pop(0)
        
    # use_time = time.time() - start
    # print("Time of posting position: %s", use_time)
    # time.sleep(0.1)
    
    yesterday_close = today_close

total_return = round(capital_record[-1]/5e8, 4)
daily_pct = pd.Series(capital_record)+5e8
daily_pct = daily_pct.pct_change()
sharpe = daily_pct.mean() / daily_pct.std() * np.sqrt(252)
sharpe = np.round(sharpe)

_name = 'v3_1_3'
plt.figure()
plt.plot(capital_record)
_title = "{}, ret:{}, sharpe:{}".format(_name, total_return, sharpe)
plt.title(_title)
plt.savefig('backtest_{}.png'.format(_name))

# factors_record = np.array(factors_record)[:,:,0]
# pd.DataFrame(factors_record).to_csv('factors_{}.csv'.format(_name))