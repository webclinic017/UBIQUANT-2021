# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import logging

from matplotlib.pyplot import close

import grpc
import question_pb2
import question_pb2_grpc
import contest_pb2
import contest_pb2_grpc

import time
import numpy as np
import pandas as pd

import math as math
from sklearn import datasets, linear_model
from scipy.optimize import minimize 
from sklearn.linear_model import LinearRegression

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from mfm.MFM import MFM

import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)


import warnings
warnings.filterwarnings("ignore")

############################ Define our factors ####################################### 
def RSIIndividual(data):
    t = 6                      # 需要6天的历史数据
    close = np.array(data[-t:])[:, :, 5].T    # N*T，所有股票的close数据（日期为最新一天）
    print("shape of one day close: ", close.shape)
    delta = np.diff(close)     # N*(T-1)

    factor = []
    for one_day in delta:
        one_day = pd.Series(one_day)
        up, down = one_day.copy(), one_day.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.ewm(span=5).mean()                      # 上涨天数的指数移动平均
        roll_down1 = down.abs().ewm(span=5).mean()            # 下跌天数的指数移动平均
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        factor.append(RSI1.iloc[-1])

    return ('RSI', factor)


def SixDayRS(data):
    '''
    六日收益率反转
    '''
    t = 6                      # 需要6天的历史数据
    close = np.array(data[-t:])[:, :, 5].T    # N*T，所有股票的close数据（日期为最新一天）
    
    factor = close[:, -1]/close[:, 0] - 1 
    return ('SixDayRS', list(factor))


def Size(data):
    '''
    市值（收盘价近似代替）
    '''
    t = 6                                   # 需要6天的历史数据
    close = np.array(data[-t:])[:, :, 5].T                 # N*T，所有股票的close数据（日期为最新一天）
    factor = -np.log(close).mean(axis=1)    # 按行求均值

    return ('Size', list(factor))


def VolumePct(data):
    '''
    成交量变化率
    '''
    t = 6                       # 需要6天的历史数据
    volume = np.array(data[-t:])[:, :, 6].T    # N*T，所有股票的volume数据（日期为最新一天）
    
    factor = []
    for one_day in volume:
        one_day = pd.Series(one_day)
        v_pct = one_day.pct_change(1)
        v_pct = v_pct.rolling(window=5).mean()
        factor.append(v_pct.iloc[-1])
    
    return ('VolumePct', factor)


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

############################ 标准化处理 #######################################
def Factor_corr(factor_values):
    
    factor_tmp = np.array([x.reshape(1, -1) for x in factor_values])   # 4 * (T*N)
    return np.corrcoef(factor_tmp)

def caculate_IC(factor_values, pct):
    factor_tmp = factor_values[:, -32:-3, :]
    factor_tmp = np.array([x.reshape(1, -1) for x in factor_values])   # 4 * (T*N)
    pct = pct[-30:-1].reshape(1, -1)                                   # 1 * (T*N)

    res = []
    for i in range(len(factor_tmp)):
        res.append(np.corrcoef(factor_tmp[i], pct)[0,1])
    return res

def initialize():
    '''
    使用login接口接入服务器
    '''
    channel = grpc.insecure_channel('47.100.97.93:40723')
    stub = contest_pb2_grpc.ContestStub(channel)
    response = stub.login(contest_pb2.LoginRequest(user_id = 67, user_pin ='GkwB5rYqHu'))
    print(response)
    return stub, response.session_key


def send_positions(_positions,_stub,_session_key,_sequence):
    '''
    提交答案。答案中包括编号sequence和一个安安数组position（position顺序和股票数据顺序一致）
    '''
    response3 = stub.submit_answer(contest_pb2.AnswerRequest(   user_id = 67, \
                                                                user_pin ='GkwB5rYqHu', \
                                                                session_key = _session_key, \
                                                                sequence = _sequence,\
                                                                positions = _positions ))
    print(response3)

def get_data(_sequence,stub2):
    response2 = stub2.get_question(question_pb2.QuestionRequest(user_id = 67, \
                                                                user_pin ='GkwB5rYqHu', \
                                                                sequence = _sequence))
    return response2

def try_to_save_time(_my_sequence, _stub2):
    '''
    初始请求时间计算，为了尽量能够更早的
    '''
    while True:
        response = get_data(_my_sequence,_stub2)
        if (_my_sequence == 0) and response.sequence>0:
            return response
        elif response.sequence == _my_sequence+1:
            return response
        else:
            # _my_sequence=0
            time.sleep(1)
            continue


# contest channel

stub,session_key = initialize()
print('connected to server...')

# question channel
channel2 = grpc.insecure_channel('47.100.97.93:40722')
stub2 = question_pb2_grpc.QuestionStub(channel2)

column_names = ['Date','Instrument','Open','High','Low','Close','Volume','Amount']
data = []

# 请求当前服务器最新的数据
my_sequence = 0
response = try_to_save_time(my_sequence, stub2)
my_sequence = response.sequence

daily_data = np.array([stock.values for stock in response.dailystk])
daily_data = daily_data[:, 0:8]
print(daily_data.shape)
data.append(daily_data)

num_of_dates = 1
factor_dates = 0

pct = []
factor_values = [[]] * 4
combo_factor_values = [[]] * 3

strategies = {'RSI':RSIIndividual, 'SixDayRS':SixDayRS, \
                  'Size':Size, 'VolumePct':VolumePct}

factor_id = {'RSI':0, 'SixDayRS':1, 'Size':2, 'VolumePct':3}
combo_factor_id = {'Mom':0, 'Size':1, 'VolumePct':2}

# 全局参数
num_of_holding = 100   # 总共持有的股票数量
leverage = 1

mp.set_start_method('fork')

while(True):
    my_sequence += 1
    response = try_to_save_time(my_sequence,stub2)
    # print(np.array(response.dailystk)[0])
    start =  time.time()
    not_end = response.has_next_question
    cur_day = response.sequence
    cur_capital = response.capital
    print("\n" + "======"*10)
    print('current day: {}, current capital: {}......\n'.format(cur_day, cur_capital))

    # daily_data = np.array(response.dailystk)
    daily_data = np.array([stock.values for stock in response.dailystk])
    daily_data = daily_data[:, 0:8]
    print(daily_data.shape)
    num_of_stock = daily_data.shape[0]

    data.append(daily_data)
    num_of_dates += 1
    # print(data)
    
    if num_of_dates <= 6:
        continue
    
    t = 6                      # 需要6天的历史数据
    close = np.array(data[-t:])[:, :, 5].T    # N*T，所有股票的close数据（日期为最新一天）
    print("shape of one day close: ", close.shape)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(task, data) for task in strategies.values()]
        for future in as_completed(futures):
            factor_name, res = future.result()
            print(factor_name, factor_id[factor_name])
            factor_values[factor_id[factor_name]].append(res)       # factor_values: k * t * n
    
    print('how many factors: ', len(factor_values))
    for i in factor_values:
        print('length of each factor: ', len(i))
        # print(i)
        print("---------------------------------------")

    close = data[-1][:, 5]        # N * 1
    cur_pct = np.diff(close)

    # final round 因子检验
    # factor_corr = data[['Mom', 'Size', 'VolumePct']].corr()
    # print(factor_corr)
    days_of_factor = len(factor_values[0])
    curr_factor = np.array(factor_values)[:, -1, :]    # k * N
    weighted_factor = curr_factor.mean(axis=0)     # N * 1
    # if days_of_factor < 30:
    #     weighted_factor = curr_factor.mean(axis=0)     # N * 1
    # else:
    #     factor_corr = Factor_corr(factor_values)
    #     factor_weight = caculate_IC(factor_values, pct)
    #     weighted_factor = curr_factor.T @ factor_weight # N * 1

    weight = [0] * num_of_stock

    _pos = np.argsort(weighted_factor)
    # print(close)
    # print(_pos)

    for k in range(0,len(_pos)):
        if _pos[k] > num_of_stock-num_of_holding/2:
            _pos[k] = 1
            weight[k] = 1/num_of_holding
        elif _pos[k] <= num_of_holding/2:
            _pos[k]= -1
            weight[k] = 1/num_of_holding
        else:
            _pos[k] = 0

    moneyspent = 0

    for k in range(0,len(_pos)):
        if close[k] == 0:
            _pos[k] = 0
        else:
            _pos[k] = int(_pos[k] * cur_capital * leverage * weight[k] / close[k])

        moneyspent += np.abs(_pos[k])*close[k]
    
    send_positions(_pos ,stub, session_key, my_sequence)
    
    print("amount of money spent:", moneyspent)
    _pos = np.array(_pos)
    stocks_to_buy = np.where(_pos>0)[0] + 6000
    stocks_to_sell = np.where(_pos<0)[0] + 6000
    # print("{} stocks to buy:".format(len(stocks_to_buy)), stocks_to_buy)
    # print("{} stocks to sell:".format(len(stocks_to_sell)), stocks_to_sell)
    
    use_time = time.time() - start
    print("Time of posting position: %s", use_time)


    use_time = time.time() - start
    print("Time of caculating IC: %s", use_time)
    time.sleep(max(4.75-use_time, 0.1))

# sys.stdout = open('logfile', 'w')
# f = open('logfile', 'w')
# backup = sys.stdout
# sys.stdout = Tee(sys.stdout, f)
# f.close()
# sys.stdout = backup    
   
