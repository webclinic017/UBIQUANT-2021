import sys
import pandas as pd
import numpy as np
sys.path.append('./')
#sys.path.append('../')

from factor_lib import *
from plot_lib import *


if __name__ == '__main__':
    init_cash = 100000
    base_path = r"../backtesting_1/"  #原始数据储存的路径
    file_name = r"CONTEST_DATA_TEST_100_1.csv"
    merged_data = mergeData(base_path, file_name)
    merged_data = calculateWeight(merged_data, equal_weight=True)
    
    # 计算因子。短均线：short_t，长均线：long_t
    short_t = 3
    long_t = 22
    
    merged_data = calculateFactor(merged_data, short_t, long_t)
    #print(merged_data.head())

    data = merged_data.set_index('Open time')
    daily_return = backTest(data)
    #daily_return = dailyBackTest(merged_data)
    #print(daily_return.head())

    ########    (1) 计算评价指标  ##########
    total_ret = daily_return['equity_curve'].values[-1]
    sharpe = calSharpe(daily_return)
    _IC = calIC(merged_data)


    print("total return: {}, sharpe ratio: {}, IC: {}, \n".format(total_ret, sharpe, _IC))

    #########   (2) 导出日志    ###########


    #########   (3) 画图    ###########
    drawCumReturn(daily_return)
    drawWeight(merged_data)
