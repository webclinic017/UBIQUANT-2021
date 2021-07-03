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
def mergeData(base_path, file_name, factor_id):
    '''
    base_path: 数据存放的路径
    file_name: 数据文件名称
    factor_id: 因子编号
    '''
    # 8个币种间的组合策略
    use_columns = [0,1,2,3,4,5,6]
    use_columns.append(factor_id+8)
    data = pd.read_csv(base_path+file_name, usecols=use_columns, 
                       names=['Open time','Instrument','Open','High','Low','Close','Volume','Alpha'])

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
    num_of_holding = param_1
    merged_data.loc[:,'Raw_factor'] = merged_data.groupby('Open time')['Alpha'].rank()
    # print(merged_data[merged_data['Instrument']==6000].head(3))
    def _Alpha100(x):
        _sell = (x.Raw_factor<=(num_of_holding/2))          ## 做多因子值最大的
        _buy = (x.Raw_factor>(500-num_of_holding/2))        ## 做空因子值最小的
        x.loc[_buy, 'Factor'] = 1
        x.loc[_sell, 'Factor'] = -1
        return x
    merged_data = merged_data.groupby('Instrument').apply(_Alpha100)
    # print('_Alpha100:\n', merged_data[merged_data['Instrument']==6000].head(5))

    merged_data.loc[:,'Factor'] = merged_data.groupby('Instrument')['Factor'].shift(2)
    merged_data['Factor'] = merged_data['Factor'].fillna(0)

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
    data['Pct_shift'] = data.groupby('Instrument')['Pct'].shift(1)
    pct_rank = data.groupby('Open time')['Pct_shift'].rank()
    normal_ic = np.corrcoef(data['Raw_factor'].fillna(0), pct_rank.fillna(0))
    return normal_ic[0][1]


