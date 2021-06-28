import sys
import pandas as pd
import numpy as np
sys.path.append('./')
#sys.path.append('../')

from factor_lib import *
from plot_lib import *


if __name__ == '__main__':
    init_cash = 100000
    time_win = 6           # n日收益率的时间窗口
    num_of_holding = 100   # 每天持有股票数量

    base_path = r"../../contest-4/"  #原始数据储存的路径
    file_name = r"CONTEST_DATA_TEST_100_1.csv"
    merged_data = mergeData(base_path, file_name)
    merged_data = calculateWeight(merged_data, num_of_holding, equal_weight=True)
    
    merged_data = calculateFactor(merged_data, time_win, num_of_holding)
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
    drawCumReturn(daily_return, (total_ret, sharpe, _IC))
    drawWeight(merged_data)
