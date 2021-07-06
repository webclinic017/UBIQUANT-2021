from matplotlib.colors import Normalize
import pandas as pd
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math as math
from sklearn import datasets, linear_model
from scipy.optimize import minimize 
from sklearn.linear_model import LinearRegression
from mfm.MFM import MFM

#In order to make the program faster, we can try to calculate the Beta first and write to CSV, and then read the csv in

############################ Define our factors ####################################### 
def RSIIndividual(close, window_length):
    '''
    相对强弱指数
    RSI = 上升平均数/(上升平均数+下跌平均数)*100%
    close.columns = ['Instrument', 'Close']
    '''
    # close = data['Close']
    delta = close.diff()
    # delta = delta.fillna(0)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=window_length, min_periods=3).mean()                # 上涨天数的指数移动平均
    roll_down1 = down.abs().ewm(span=window_length, min_periods=3).mean()            # 下跌天数的指数移动平均
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    return -RSI1


def SixDayRS(close, window_length: int):
    '''
    六日收益率反转
    '''
    rs = -close.pct_change(6)
    return rs

def OpenCloseDiff(OCDiff, window_length):
    '''
    OCDiff指数平均
    '''
    OCDiff = np.abs(OCDiff)
    OCDiff = OCDiff.rolling(window=window_length, min_periods=3).mean()
    return -OCDiff

def Size(price, window_length):
    '''
    OCDiff指数平均
    '''
    price = np.log(price)
    price = price.rolling(window=window_length, min_periods=3).mean()
    return -price


def F_Volume(volume, window_length):
    '''
    OCDiff指数平均
    '''
    volume = np.log(volume)
    volume = volume.rolling(window=window_length, min_periods=3).mean()
    return -volume


def VolumePct(volume, window_length):
    '''
    OCDiff指数平均
    '''
    v_pct = volume.pct_change(3)
    v_pct = v_pct.rolling(window=window_length, min_periods=3).mean()
    return -v_pct

def Volatility(close, window_length):
    pct = close.pct_change()
    volatility = pct.rolling(window=window_length, min_periods=21).std()
    return -volatility

############################ 去极值处理 #######################################
# def Winsorize(factor: pd.Series, n=2):
#     '''
#     MAD方法，默认n=2
#     '''
#     median = factor.expanding().median()                   # cummedian
#     MAD = (np.abs(factor) - median).expanding().median()
#     factor[factor>median+n*MAD] = median+n*MAD             # 剔除偏离中位数5倍以上的数据
#     factor[factor<median-n*MAD] = median-n*MAD
#     return factor
def Winsorize(factor: pd.Series, n=2):
    '''
    MAD方法，默认n=2
    '''
    median = np.median(factor)
    MAD = np.median((np.abs(factor) - median))
    factor[factor>median+n*MAD] = median+n*MAD             # 剔除偏离中位数5倍以上的数据
    factor[factor<median-n*MAD] = median-n*MAD
    return factor

############################ 标准化处理 #######################################
def Standardlize(factor):
    '''
    横截面上z-score标准化
    '''
    factor = (factor-factor.mean())/factor.std()
    return factor

############################ 因子中性化处理 #######################################
# 行业中性化处理和市值中性化处理
def SizeNeutral(df, factor):
    industry_factor = [i for i in df.columns if i[:15]=='industry_prefix']
    # 对市值因子和行业因子进行回归
    xvars = ['market_cap_float'] + industry_factor
    used_factors = xvars + [factor] + ['instrument']
    
    used_factors_df = df[used_factors]
    used_factors_df = used_factors_df[~used_factors_df.isnull().any(axis=1)]
    if len(used_factors_df) == 0:
        return None
    X = used_factors_df[xvars]
    y = used_factors_df[factor]
    reg = LinearRegression()
    try:
        reg.fit(X,y)   # 将行业因子和市值因子对特定因子作回归
        res = y-reg.predict(X)
        used_factors_df[factor] = res
    except ValueError as e:
        used_factors_df[factor] = np.nan 

    return used_factors_df[['instrument',factor]]

if __name__ == "__main__":
    base_path = r"../"  #原始数据储存的路径
    file_name = r"CONTEST_DATA_TEST_100_1.csv"
    data = pd.read_csv(base_path+file_name, 
                       usecols=[0,1,2,3,4,5,6,7], 
                       names=['Date','Instrument','Open','High','Low','Close','Volume','Amount'])

    # 计算因子
    strategies = {'RSI':RSIIndividual, 'SixDayRS':SixDayRS, 'OCDiff':OpenCloseDiff, \
                  'Size':Size, 'F_Volume':F_Volume, 'VolumePct':VolumePct, 'Volatility':Volatility}
    # factor_name = 'Volatility'
    # strategy = strategies[factor_name]

    data.loc[:, 'RSI'] = data.groupby('Instrument')['Close'].apply(RSIIndividual, (5))
    data.loc[:,'RSI'] = data.groupby('Instrument')['RSI'].shift(2)
    data.loc[:, 'RSI'] = data.groupby('Date')['RSI'].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, 'RSI'] = data.groupby('Date')['RSI'].apply(Standardlize)            # 因子标准化处理

    data.loc[:, 'SixDayRS'] = data.groupby('Instrument')['Close'].apply(SixDayRS, (5))
    data.loc[:,'SixDayRS'] = data.groupby('Instrument')['SixDayRS'].shift(2)
    data.loc[:, 'SixDayRS'] = data.groupby('Date')['SixDayRS'].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, 'SixDayRS'] = data.groupby('Date')['SixDayRS'].apply(Standardlize)            # 因子标准化处理

    data.loc[:, 'OCDiff'] = data['Open']/data.groupby('Instrument')['Close'].shift(1)-1
    data.loc[:, 'OCDiff'] = data.groupby('Instrument')['OCDiff'].apply(OpenCloseDiff, (5))
    data.loc[:,'OCDiff'] = data.groupby('Instrument')['OCDiff'].shift(2)
    data.loc[:, 'OCDiff'] = data.groupby('Date')['OCDiff'].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, 'OCDiff'] = data.groupby('Date')['OCDiff'].apply(Standardlize)            # 因子标准化处理

    data.loc[:, 'Size'] = data.groupby('Instrument')['Close'].apply(Size, (5))
    data.loc[:,'Size'] = data.groupby('Instrument')['Size'].shift(2)
    data.loc[:, 'Size'] = data.groupby('Date')['Size'].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, 'Size'] = data.groupby('Date')['Size'].apply(Standardlize) 

    data.loc[:, 'F_Volume'] = data.groupby('Instrument')['Volume'].apply(F_Volume, (5))
    data.loc[:,'F_Volume'] = data.groupby('Instrument')['F_Volume'].shift(2)
    data.loc[:, 'F_Volume'] = data.groupby('Date')['F_Volume'].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, 'F_Volume'] = data.groupby('Date')['F_Volume'].apply(Standardlize) 

    data.loc[:, 'VolumePct'] = data.groupby('Instrument')['Volume'].apply(VolumePct, (5))
    data.loc[:,'VolumePct'] = data.groupby('Instrument')['VolumePct'].shift(2)
    data.loc[:, 'VolumePct'] = data.groupby('Date')['VolumePct'].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, 'VolumePct'] = data.groupby('Date')['VolumePct'].apply(Standardlize) 

    data.loc[:, 'Volatility'] = data.groupby('Instrument')['Close'].apply(Volatility, (30))
    data.loc[:,'Volatility'] = data.groupby('Instrument')['Volatility'].shift(2)
    data.loc[:, 'Volatility'] = data.groupby('Date')['Volatility'].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, 'Volatility'] = data.groupby('Date')['Volatility'].apply(Standardlize) 

    print(data[data['Instrument']==6000])
    print(data[data['Instrument']==6499])

    # 因子相关性
    all_factors = list(strategies.keys())
    factor_corr = data[all_factors].corr()
    print(factor_corr)

    # RSI和SixDayRS大类因子合成
    data.loc[:,'Mom'] = (data.loc[:,'RSI'] + data.loc[:,'SixDayRS'])/2
    data.drop(['RSI', 'SixDayRS'], axis=1, inplace=True)

    # 去除无效因子OCDiff、F_Volume、Volatility无效因子
    data.drop(['OCDiff', 'F_Volume', 'Volatility'], axis=1, inplace=True)

    # final round 因子检验
    factor_corr = data[['Mom', 'Size', 'VolumePct']].corr()
    print(factor_corr)

    # 构造多因子模型（不知道是不是Barra）


    # 资产的日收益率
    data.loc[:,'Pct'] = data.groupby('Instrument')['Close'].pct_change(1)

    # 测试因子
    factor_name = 'Equal'
    group = 10
    pnl_record = []
    sorted_dates = np.sort(np.unique(data['Date'].values))
    T = len(sorted_dates) 

    if factor_name == 'Equal':
        for i in range(T):
            daily_data = data[data['Date']==sorted_dates[i]]
            if np.sum(daily_data[['Mom', 'Size', 'VolumePct']].isnull().any())>0:
                continue
            
            N = daily_data.shape[0]
            equal_factor = (daily_data['Mom']+daily_data['Size']+daily_data['VolumePct'])/3

            _rank = equal_factor.rank()
            # print(_rank)
            _buy = _rank>int((N*9/10))
            # print(_buy)
            _sell = _rank<int((N/10))

            weight = 1/N/2*10
            # print(weight)
            # print(daily_data.loc[_buy, 'Pct'])
            daily_return = np.sum(daily_data.loc[_buy, 'Pct'])*weight - np.sum(daily_data.loc[_sell, 'Pct'])*weight
            # print(daily_return)

            pnl_record.append(daily_return)
            
    else:
        for date in sorted_dates:
            daily_data = data[data['Date']==date]
            # print('date: ', date)
            # print(daily_data)
            # print(daily_data.loc[:,factor_name].isnull().any())
            if daily_data[factor_name].isnull().any():
                continue
            num_of_stock = daily_data.shape[0]
            _rank = daily_data.loc[:,factor_name].rank()
            # print(_rank)
            _buy = _rank>int((num_of_stock*9/10))
            # print(_buy)
            _sell = _rank<int((num_of_stock/10))

            weight = 1/num_of_stock/2*10
            # print(weight)
            # print(daily_data.loc[_buy, 'Pct'])
            daily_return = np.sum(daily_data.loc[_buy, 'Pct'])*weight - np.sum(daily_data.loc[_sell, 'Pct'])*weight
            # print(daily_return)

            pnl_record.append(daily_return)
        
    
    pnl = np.cumprod(1+np.array(pnl_record))-1
    print('strategy: ', factor_name, '. total return: ', pnl[-1])
    
    # 画图
    plt.figure()
    plt.plot(pnl)
    # plt.savefig(factor_name+'.png')
    plt.savefig('./multifactors_figure/{}_pnl.png'.format(factor_name))
