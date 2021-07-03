import sys
import os
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append('./')
#sys.path.append('../')

from factor_lib import *
from plot_lib import *


def backtesting(factor_num):
    init_cash = 100000
    time_win = 5             # 隔夜跳空的n日平均值
    num_of_holding = 80      # 每天持有股票数量

    base_path = r"../"  #原始数据储存的路径
    file_name = r"CONTEST_DATA_TEST_100_1.csv"

    merged_data = mergeData(base_path, file_name, factor_id=factor_num)
    merged_data = calculateWeight(merged_data, num_of_holding, equal_weight=True)
    merged_data = calculateFactor(merged_data, num_of_holding)
    

    data = merged_data.set_index('Open time')
    daily_return = backTest(data)
    #daily_return = dailyBackTest(merged_data)
    #print(daily_return.head())

    ########    (1) 计算评价指标  ##########
    total_ret = daily_return['equity_curve'].values[-1]
    sharpe = calSharpe(daily_return)
    _IC = calIC(merged_data)

    result = "factor: {}, total return: {}, sharpe ratio: {}, IC: {}, \n".format(factor_num, total_ret, sharpe, _IC)
    print(result)

    #########   (3) 画图    ###########
    drawCumReturn(daily_return, (factor_num, total_ret, sharpe, _IC))

    return result

def _batch(factor_zoo):
    batch_size = len(factor_zoo)//os.cpu_count()
    #print("batch_size: ", batch_size)
    for i in range(0, len(factor_zoo), batch_size):
        yield factor_zoo[i:i+batch_size]

def _mp(batch):
    result = ''
    for i in batch:
        tmp = backtesting(i)
        result += tmp
        time.sleep(1)
    
    return result
        

if __name__ == "__main__":
    factor_zoo = np.arange(100)
    #print(factor_zoo)
    factor_batchs = list(_batch(factor_zoo))
    #print(factor_batchs)

    with open('test_alpha100.txt', 'w+') as f:
        with ProcessPoolExecutor() as excutor:
            futures = [excutor.submit(_mp, batch) for batch in factor_batchs]
            for future in as_completed(futures):
                result = future.result()
                f.write(result)
                f.flush()
    




