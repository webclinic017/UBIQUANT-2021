## -*- coding: utf-8 -*-
#  定于画图相关的函数

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#################    Part1 策略净值图    #################
def drawCumReturn(data, result, file_name='策略净值图.png'):
    if data.index.name != 'Open time':
        data = data.set_index('Open time')

    # figure = plt.figure()
    # plt.figure(figsize = (15,7))
    # print(data['equity_curve'])
    
    plt.figure()               
    plt.plot(data['equity_curve'], label='equity_curve', c='red')
    factor_num, total_ret, sharpe, _IC = result
    plt.title("factor: {}, ret: {}, sharpe: {}, IC: {}, \n".format(factor_num, round(total_ret,2), round(sharpe,2), round(_IC,4)))
    plt.legend()
    file_name = "factor_{}.png".format(factor_num)
    plt.savefig(file_name)



def drawWeight(data, file_name='资金权重.png'):
    if data.index.name != 'Open time':
        data = data.set_index('Open time')

    plt.figure(figsize = (15,8))
    for instrument, group in data.groupby('Instrument'):
        plt.plot(group['Weight'], label=instrument)

    #显示标签
    plt.legend()
    #画图
    plt.savefig(file_name)
