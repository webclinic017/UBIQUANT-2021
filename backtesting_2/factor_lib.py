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
def calculateWeight(data, equal_weight=True):
    debug = False

    # 资产日收益率的波动率,过去100天（衡量资产的历史风险水平）
    data.loc[:,'Volatility'] = data.groupby('Instrument')['Pct'].transform(lambda x: x.rolling(100).std())
    data.loc[:,'Volatility'] = data.groupby('Instrument')['Volatility'].shift(2)
    # 股票数量
    num_of_stock = len(pd.unique(data['Instrument']))
    # 横截面上根据波动率倒数分配权重
    data.loc[:,'Weight'] = 1/num_of_stock

    return data



#################    Part3 计算因子    #################
def calculateFactor(merged_data, param_1=None, param_2=None, param_3=None, param_4=None):
    '''
    双均线策略(根据收盘价计算)
    param_1: short_t: 短均线周期
    param_2: long_t: 长均线周期
    '''
    short_t = param_1
    long_t = param_2

    # Signal 1:利用短期移动均线与长期移动均线的交叉来择时
    merged_data.loc[:,'Signal_1'] = 0
    merged_data.loc[:,'Sma'] = merged_data.groupby('Instrument')['Close'].transform(lambda x: x.rolling(short_t).mean())
    merged_data.loc[:,'Lma'] = merged_data.groupby('Instrument')['Close'].transform(lambda x: x.rolling(long_t).mean())
    def MA(x):
        _buy = (x.Sma>x.Sma.shift()) & (x.Sma>x.Lma) & (x.Sma.shift()<x.Lma.shift())  ## 买入信号
        _sell = (x.Lma<x.Lma.shift()) & (x.Sma<x.Lma) & (x.Sma.shift()>x.Lma.shift()) ## 卖出信号
        x.loc[_buy, 'Signal_1'] = 1
        x.loc[_sell, 'Signal_1'] = -1
        return x
    merged_data = merged_data.groupby('Instrument').apply(MA)
    # print('MA:\n', merged_data)

    # Signal 2:MACD择时
    merged_data.loc[:, 'Signal_2'] = 0
    merged_data.loc[:, 'DIFF'] = merged_data['Sma']-merged_data['Lma']
    merged_data.loc[:, 'DEA'] = merged_data.groupby('Instrument')['DIFF'].transform(lambda x: x.rolling(short_t).mean())
    def MACD(x):
        _buy = (x.DIFF>x.DIFF.shift()) & (x.DIFF>x.DEA) & (x.DIFF.shift()<x.DEA.shift()) & (x.DIFF>0)
        _sell = (x.DIFF<x.DIFF.shift()) & (x.DIFF<x.DEA) & (x.DIFF.shift()>x.DEA.shift()) & (x.DIFF<0)
        x.loc[_buy, 'Signal_2'] = 1
        x.loc[_sell, 'Signal_2'] = -1
        return x
    merged_data = merged_data.groupby('Instrument').apply(MACD)

    # Signal 3:TRIX择时
    merged_data.loc[:, 'Signal_3'] = 0
    merged_data.loc[:, 'EMA'] = merged_data.groupby('Instrument')['Sma'].transform(lambda x: x.rolling(short_t).mean())
    merged_data.loc[:, 'EMA'] = merged_data.groupby('Instrument')['EMA'].transform(lambda x: x.rolling(short_t).mean())
    merged_data.loc[:, 'TRIX'] = merged_data.groupby('Instrument')['EMA'].pct_change(1)
    merged_data.loc[:, 'MATRIX'] = merged_data.groupby('Instrument')['TRIX'].transform(lambda x: x.rolling(short_t).mean())
    def TRIX(x):
        _buy = (x.TRIX>x.TRIX.shift()) & (x.TRIX>x.MATRIX) & (x.TRIX.shift()<x.MATRIX.shift())
        _sell = (x.TRIX<x.TRIX.shift()) & (x.TRIX<x.MATRIX) & (x.TRIX.shift()>x.MATRIX.shift())
        x.loc[_buy, 'Signal_3'] = 1
        x.loc[_sell, 'Signal_3'] = -1
        return x
    merged_data = merged_data.groupby('Instrument').apply(TRIX)

    # 因子原始值
    merged_data.loc[:,'Raw_factor'] = merged_data['Signal_1']+merged_data['Signal_2']+merged_data['Signal_3']
    merged_data.loc[:,'Flag'] = None
    merged_data.loc[:,'Factor'] = None
    def factor(x):
        _buy = x.Raw_factor>0
        _sell = x.Raw_factor<-1
        x.loc[_buy, 'Flag'] = 1
        x.loc[_buy, 'Factor'] = 1
        x.loc[_sell, 'Flag'] = -1
        x.loc[_sell, 'Factor'] = -1
        return x
    # t日产生信号，t+1建仓，t+2开始确定收益；因子值向下填充
    merged_data = merged_data.groupby('Instrument').apply(factor)
    merged_data.loc[:,'Flag'] = merged_data.groupby('Instrument')['Flag'].shift(2)
    merged_data.loc[:,'Factor'] = merged_data.groupby('Instrument')['Factor'].ffill()
    merged_data.loc[:,'Factor'] = merged_data.groupby('Instrument')['Factor'].shift(2)
    # merged_data['Factor'].fillna(0)

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
    # 计算各个币种的涨幅，检查是否有爆仓
    each_cum_return = pd.DataFrame()
    for column in cum_return.columns:
        each_cum_return.loc[:,column] = cum_return[column].cumprod()
    
    daily_return.loc[:,'strategy_mean'] = daily_return.sum(axis=1)
    daily_return.loc[:,'equity_curve'] = (daily_return['strategy_mean']+1).cumprod()

    btc_pct = data[data['Instrument']=='BTCUSDT']
    daily_return = daily_return.join(btc_pct['Pct'], how='outer')

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

    tmp = data.set_index('Open time')
    btc_pct = tmp[tmp['Instrument']=='BTCUSDT']
    daily_return = daily_return.join(btc_pct['Pct'], how='outer')
    daily_return.loc[:,'btc_curve'] = (1+daily_return['Pct']).cumprod()

    #（equity_curve:美元衡量的绝对收益, relative_equity_curve:相对于btc的相对收益）
    daily_return.loc[:,'relative_equity_curve'] = daily_return['equity_curve']/daily_return['btc_curve']
    
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
    normal_ic = np.corrcoef(data['Raw_factor'].fillna(0), data['Pct'].fillna(0))
    return normal_ic[0][1]


