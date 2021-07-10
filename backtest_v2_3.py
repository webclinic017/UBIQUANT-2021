'''
v_2_4
第一个模拟真实比赛环境的回测版本
四因子 + Alpha101，全部找下来有十几个因子比较有效
但是还没验证因子之间的相关性，先对所有因子计算一个史密斯正交化分解试试
再搞一个IC加权
因子等权重组合
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
def RSIIndividual(data):
    t = 6                           # 需要6天的历史数据
    close = data[:, 5]    # 股票N天的close数据
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
    # return (-RSI1.iloc[-1], _index)


def SixDayRS(data):
    '''
    六日收益率反转
    '''
    t = 6                           # 需要6天的历史数据
    close = data[:, 5]    # 股票N天的close数据    
    # return (-(close[-1]/close[-t] - 1), _index)
    return -(close[-1]/close[-t] - 1)


def Size(data):
    '''
    市值（收盘价近似代替）
    '''
    t = 6                                   # 需要6天的历史数据
    close = data[:, 5]            # 股票N天的close数据
    size = np.log(close)         # 按行求均值
    size = pd.Series(size)
    size = size.rolling(window=5).mean()

    return -size.iloc[-1]


def VolumePct(data):
    '''
    成交量变化率
    '''
    t = 6                                      # 需要6天的历史数据
    volume = data[:, 6]              # 股票N天的volume数据
    
    volume = pd.Series(volume)
    # v_pct = np.abs(volume.pct_change(1))
    v_pct = volume.pct_change(3)
    v_pct = v_pct.rolling(window=5).mean()
    return -v_pct.iloc[-1]

###############################   Alpha 101   ##############################
def Alpha6(data):
    _open = pd.Series(data[:, 2])                # 开盘价
    volume = pd.Series(data[:, 6])              # 股票N天的volume数据
    alpha6 = _open.rolling(window=10).corr(volume)
    return -1*alpha6.iloc[-1]

def Alpha7(data):
    close = pd.Series(data[:, 5])
    amount = pd.Series(data[:, 7])              # 股票N天的amount数据
    mean_amount = amount.rolling(window=5).mean().iloc[-1]
    if mean_amount<amount.iloc[-1]:
        delta = close.diff(3).abs().iloc[-10:]
        delta_rank = delta.rank()
        return -1*delta_rank.iloc[-1]*np.sign(delta.iloc[-1])
    else:
        return -1

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
    close = pd.Series(data[:, 5])
    amount = pd.Series(data[:, 7])              # 股票N天的amount数据
    _sign = np.sign(amount.diff(1).iloc[-1])
    delta = close.diff(1)
    return -1*_sign*delta.iloc[-1]

def Alpha23(data):
    high = pd.Series(data[:, 3])
    _mean = high.rolling(10).mean()
    _delta = high.diff(2)
    if _mean.iloc[-1]<high.iloc[-1]:
        return -1*_delta.iloc[-1]
    else:
        return 0

def Alpha26(data):
    high = pd.Series(data[:, 3])[-10:]
    amount = pd.Series(data[:, 7])[-10:]
    high_rank = high.rank()
    amount_rank = amount.rank()
    _corr = amount_rank.rolling(window=5).corr(high_rank)
    res = _corr.rolling(window=3).max()

    return -1*res.iloc[-1]

def Alpha28(data):
    high = pd.Series(data[:, 3])
    low = pd.Series(data[:, 4])
    close = pd.Series(data[:, 5])
    amount = pd.Series(data[:, 7])
    adv10 = amount.rolling(window=10).mean().fillna(0)
    _corr = adv10.rolling(window=5).corr(low).fillna(0)
    _scale = _corr+(high+low)/2-close
    _scale /= _scale.abs().sum()
    
    return _scale.iloc[-1]

def Alpha41(data):
    high = data[:, 3][-1]
    low = data[:, 4][-1]
    volume = data[:, 6][-1]
    amount = data[:, 7][-1]

    return np.sqrt(high*low) - amount/volume

def Alpha43(data):
    close = pd.Series(data[:, 5])
    amount = pd.Series(data[:, 7])
    adv10 = amount.rolling(10).mean()
    amount_rank = (amount/adv10).rank()
    delta = (-1*close.diff(4)).iloc[-4:]
    delta_rank = delta.rank()

    return amount_rank.iloc[-1]*delta_rank.iloc[-1]

def Alpha46(data):
    close = data[:, 5]
    condition1 = (close[-20]-close[-10])/10 - (close[-10]-close[-1])/10
    if condition1 > 0.25:
        return -1
    elif condition1 < 0:
        return 1
    else:
        return -1*(close[-1]-close[-2])

def Alpha49(data):
    close = data[:, 5]
    condition1 = (close[-20]-close[-10])/10 - (close[-10]-close[-1])/10
    if condition1 < -0.1:
        return 1
    else:
        return -1*(close[-1]-close[-2])

def Alpha51(data):
    close = data[:, 5]
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
    strategy_list = [RSIIndividual, SixDayRS, Size, VolumePct, Alpha6, Alpha9, \
                     Alpha12, Alpha23, Alpha26, Alpha41, Alpha46, Alpha49, Alpha51, \
                     Alpha53, Alpha54]
    num_of_strategies = len(strategy_list)

    factors = []
    # factors = [0] * num_of_strategies
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(strategy_list[i], _array, i) for i in range(num_of_strategies)]
    #     for future in as_completed(futures):
    #         factor, _index = future.result()
    #         factors[_index] = factor
    
    _array = np.array(_array)
    factors.append(RSIIndividual(_array))
    factors.append(SixDayRS(_array))
    factors.append(Size(_array))
    factors.append(VolumePct(_array))
    factors.append(Alpha6(_array))
    factors.append(Alpha9(_array))
    factors.append(Alpha12(_array))
    factors.append(Alpha23(_array))
    factors.append(Alpha23(_array))
    factors.append(Alpha26(_array))
    factors.append(Alpha41(_array))
    factors.append(Alpha46(_array))
    factors.append(Alpha49(_array))
    factors.append(Alpha51(_array))
    factors.append(Alpha53(_array))
    factors.append(Alpha54(_array))
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
def Schimidt(factors):    
    R = np.zeros((factors.shape[1], factors.shape[1]))
    Q = np.zeros(factors.shape)
    for k in range(0, factors.shape[1]):
        R[k, k] = np.sqrt(np.dot(factors[:, k], factors[:, k]))
        Q[:, k] = factors[:, k]/R[k, k]
        for j in range(k+1, factors.shape[1]):
            R[k, j] = np.dot(Q[:, k], factors[:, j])
            factors[:, j] = factors[:, j] - R[k, j]*Q[:, k]

    return Q

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

    if len(data[0]) >= 20:

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
        all_factors = Schimidt(all_factors)                                   # Schimidt正交化分解
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

        factors_record_len = len(factors_record)
        if factors_record_len >= 10:
            # 其实只做了8次回归，取平均值
            factors_weight = []
            for i in range(factors_record_len-2):
                factors_tmp = factors_record[-(i+3)]                    # N * K
                pct_tmp = pct_record[-(i+1)]                            # N * 1
                factors_corr = np.cov(factors_tmp, rowvar=0)
                factors_pct_corr = [np.cov(factors_tmp[:, i], pct_tmp)[0,1] for i in range(num_of_factor)]
                factors_weight.append(factors_corr @ factors_pct_corr)
            factors_weight = np.array(factors_weight).mean(axis=0)      # 按列平均
            # with ProcessPoolExecutor() as executor:
            #     futures = [executor.submit(factor_regression, factors_record, pct_record, i+1) for i in range(len(factors_record)-2)]
            #     for future in as_completed(futures):
            #         factors_weight.append(future.result())
            
            # factors_weight = np.array(factors_weight).mean(axis=0)
            # factors_weight /= np.sum(factors_weight)

            # print("factors_weight: ", factors_weight)
            if factors_record_len >= 10: factors_record.pop(0)

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

_name = 'v2_4'
plt.figure()
plt.plot(capital_record)
_title = "{}, ret:{}, sharpe:{}".format(_name, total_return, sharpe)
plt.title(_title)
plt.savefig('backtest_{}.png'.format(_name))

# factors_record = np.array(factors_record)[:,:,0]
# pd.DataFrame(factors_record).to_csv('factors_{}.csv'.format(_name))