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

#In order to make the program faster, we can try to calculate the Beta first and write to CSV, and then read the csv in

############################ Define our factors ####################################### 
def RSIIndividual(close: pd.Series, window_length: int):
    '''
    相对强弱指数
    RSI = 上升平均数/(上升平均数+下跌平均数)*100%
    close.columns = ['Instrument', 'Close']
    '''
    delta = close.diff()
    # delta = delta.fillna(0)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=window_length, min_periods=3).mean()                # 上涨天数的指数移动平均
    roll_down1 = down.abs().ewm(span=window_length, min_periods=3).mean()            # 下跌天数的指数移动平均
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    return RSI1


#Try some new factors
def PricetoLowest(data):
    close = data.loc[:, 'close']
    stock = data.loc[:, 'instrument']
    RatiotoLowest = pd.Series()
    for i in list(range(1,len(stock)+1)):
        RatiotoLowest = RatiotoLowest.append(pd.Series((close.iloc[np.shape(close)[0]-1][[stock[i-1]]]/min(close[stock[i-1]]))))
    returnv = (RatiotoLowest)
    FactorValue = returnv
    FactorValue = (FactorValue - np.mean(FactorValue))/np.std(FactorValue)
    return FactorValue


############################ 去极值处理 #######################################
def Winsorize(factor: pd.Series, n=2):
    '''
    MAD方法，默认n=2
    '''
    median = factor.expanding().median()                   # cummedian
    MAD = (np.abs(factor) - median).expanding().median()
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
    factor_name = 'RSI'
    data.loc[:, factor_name] = data.groupby('Instrument')['Close'].apply(RSIIndividual, (4))
    data.loc[:,factor_name] = data.groupby('Instrument')[factor_name].shift(2)                  # 预测t+2日的收益
    # data.fillna(0)

    data.loc[:, factor_name] = data.groupby('Instrument')[factor_name].apply(Winsorize, (2))    # 因子离群值处理
    data.loc[:, factor_name] = data.groupby('Date')[factor_name].apply(Standardlize)            # 因子标准化处理

    # print(data[data['Instrument']==6000])
    # print(data[data['Instrument']==6001])

    # 资产的日收益率
    data.loc[:,'Pct'] = data.groupby('Instrument')['Close'].pct_change(1)

    group = 10
    pnl_record = []
    data_list = np.sort(np.unique(data['Date'].values))

    for date in data_list:
        daily_data = data[data['Date']==date]
        # print('date: ', date)
        # print(daily_data[factor_name].isnull().any())
        if daily_data[factor_name].isnull().any():
            continue
        num_of_stock = daily_data.shape[0]
        _rank = daily_data[factor_name].rank()
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
    print('total return: ', pnl[-1])
    
    # 画图
    plt.figure()
    plt.plot(pnl)
    # plt.savefig(factor_name+'.png')
    plt.savefig(factor_name+'_standardized.png')
