## -*- coding: utf-8 -*-
#  定于计算因子相关的函数

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import os

import warnings
warnings.filterwarnings('ignore')

#################    Part1 导入、合并数据    #################
def mergeData(base_path, file_name):
    '''
    base_path: 数据存放的路径
    '''
    # 8个币种间的组合策略
    data = pd.read_csv(base_path+file_name, usecols=[0,1,2,3,4,5,6],names=['Open time','Instrument','Open','High','Low','Close','Volume'])

    # 资产的日收益率
    data.loc[:,'Pct'] = data.groupby('Instrument')['Close'].pct_change(1)

    return data

#################    Part2 计算各个币种的资金权重    #################
def calculateWeight(data, num_of_holding, equal_weight=True):
    debug = False

    # 资产日收益率的波动率,过去100天（衡量资产的历史风险水平）
    # data.loc[:,'Volatility'] = data.groupby('Instrument')['Pct'].transform(lambda x: x.rolling(100).std())
    # data.loc[:,'Volatility'] = data.groupby('Instrument')['Volatility'].shift(2)
    # 股票数量
    num_of_stock = len(pd.unique(data['Instrument']))
    # 横截面上根据波动率倒数分配权重
    data.loc[:,'Weight'] = 1/num_of_holding

    return data



#################    Part3 计算因子    #################
def calculateFactor(merged_data, param_1=None, param_2=None, param_3=None, param_4=None):
    '''
    股价因子
    param_1: time_window: 股价的n日平均值
    '''
    time_window = param_1
    num_of_holding = param_2
    
    merged_data.loc[:,'Factor'] = 0
    
    merged_data.loc[:,'Volume_MA'] = merged_data.groupby('Instrument')['Volume'].transform(lambda x: x.rolling(time_window).mean())
    
    merged_data.loc[:,'Raw_factor'] = merged_data.groupby('Open time')['Volume_MA'].rank()
    print(merged_data[merged_data['Instrument']==6000].head(3))
    def RS(x):
        _buy = (x.Raw_factor<=(num_of_holding/2))          ## 做多成交量最低的
        _sell = (x.Raw_factor>(500-num_of_holding/2))      ## 做空成交量最大的
        x.loc[_buy, 'Factor'] = 1
        x.loc[_sell, 'Factor'] = -1
        return x
    merged_data = merged_data.groupby('Instrument').apply(RS)
    print('RS:\n', merged_data[merged_data['Instrument']==6000].head(5))

    merged_data.loc[:,'Factor'] = merged_data.groupby('Instrument')['Factor'].shift(2)
    merged_data['Factor'].fillna(0)

    return merged_data

#################    Part4-1 计算策略收益    #################
def backTest(data):

    instrument_list = set(data['Instrument'])
    count = 0
    for instrument in instrument_list:
        df_tmp = data[data['Instrument']==instrument]
        #tmp = (df_tmp['Pct']*df_tmp['Factor']+1)*(df_tmp['Buy_at_open']*df_tmp['Flag']+1)*df_tmp['Weight']
        tmp = (df_tmp['Pct']*df_tmp['Factor']*df_tmp['Weight']+1)
        tmp.name = instrument
        #print(instrument, '\n', tmp)
        if count == 0:
            # daily_return是日收益率
            daily_return = pd.DataFrame(tmp-1)
            # cum_return是+1之后的日涨跌幅
            cum_return = pd.DataFrame(tmp)
        else:
            daily_return = daily_return.join(tmp-1, how='outer')
            cum_return = cum_return.join(tmp, how='outer')
        count += 1
    #print('daily return\n', daily_return)
    
    each_cum_return = pd.DataFrame()
    for column in cum_return.columns:
        each_cum_return.loc[:,column] = cum_return[column].cumprod()
    
    daily_return.loc[:,'strategy_mean'] = daily_return.sum(axis=1)
    daily_return.loc[:,'equity_curve'] = (daily_return['strategy_mean']+1).cumprod()


    return daily_return

#################    Part4-2 每日循环的方式计算策略收益    #################
def dailyBackTest(data):
    '''
    highPrice,lowPrice,closePrice: m*n维矩阵，m代表数据量（日频数据），n代表资产数量
    weights: 横截面上各个币种分配的资金数量。m*n维矩阵，m代表数据量（日频数据），n代表资产数量
    factors: 各个币种的因子值，+1建多仓，-1建空仓，0空仓。m*n维矩阵，m代表数据量（日频数据），n代表资产数量
    '''
    # 将所需变量存进二维数组
    highPrice = pd.pivot(data, index='Open time', columns='Instrument', values='High').values
    lowPrice = pd.pivot(data, index='Open time', columns='Instrument', values='Low').values
    closePrice = pd.pivot(data, index='Open time', columns='Instrument', values='Close').values
    weights = pd.pivot(data, index='Open time', columns='Instrument', values='Weight').values
    factors = pd.pivot(data, index='Open time', columns='Instrument', values='Factor').values
    df_close = pd.pivot(data, index='Open time', columns='Instrument', values='Close')

    #---------------------------------策略参数-------------------------------
    cash = 100000
    cash_record = [100000]
    #sec = 0.12                #保证金比率
    #commission = 0.0001       #手续费
    #slip = 0                  #滑点

    daily_profit = []

    highPrice = np.array(highPrice)
    lowPrice = np.array(lowPrice)
    closePrice = np.array(closePrice)
    weights = np.array(weights)
    factors = np.array(factors)
    #print(factors)

    for i in range(0,len(closePrice)):
    #############################################      策略主体：每日调仓     ########################################### 
        #print("weights:\n", weights[i])
        #print("close price:\n", closePrice[i])
        
        if i==0:
            one_day_profit = [0]*len(closePrice[i])
            daily_profit.append(list(one_day_profit))
            #print("position record:\n", position_record)
        else:
            #one_day_profit = np.multiply(weights[i], np.divide(closePrice[i], closePrice[i-1])-1)
            one_day_profit = np.multiply(np.multiply(factors[i], weights[i]), np.divide(closePrice[i], closePrice[i-1])-1)
            daily_profit.append(list(one_day_profit))

    #########################################################################################################
    daily_return = pd.DataFrame(daily_profit, index=df_close.index, columns=df_close.columns)
    
    daily_return.loc[:,'strategy_mean'] = daily_return.sum(axis=1)
    daily_return.loc[:,'equity_curve'] = (daily_return['strategy_mean']+1).cumprod()
    
    #print(daily_return)
    return daily_return    


def calSharpe(data):
    '''
    计算夏普比率
    '''
    data.loc[:,'strategy_mean'] = data['equity_curve'].pct_change().fillna(0)
    sharpe = np.mean(data['strategy_mean']-0.04/250)/np.std(data['strategy_mean'])  # 日平均收益减去无风险收益率
    sharpe *= np.sqrt(250)

    return sharpe

def calIC(data):
    #print(data)
    normal_ic = np.corrcoef(data['Raw_factor'].fillna(0), data['Pct'].shift(3).fillna(0))
    return normal_ic[0][1]


