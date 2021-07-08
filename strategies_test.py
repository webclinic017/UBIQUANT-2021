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

import os
import time
import numpy as np
import pandas as pd

import math as math
from sklearn import preprocessing
from scipy.optimize import minimize 
from sklearn.linear_model import LinearRegression

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from mfm.MFM import MFM


import warnings
warnings.filterwarnings("ignore")

def initialize():
    '''
    使用login接口接入服务器
    '''
    channel = grpc.insecure_channel('47.100.97.93:40723')
    stub = contest_pb2_grpc.ContestStub(channel)
    response = stub.login(contest_pb2.LoginRequest(user_id = 67, user_pin ='GkwB5rYqHu'))
    # print(response)
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
        
        if response.sequence == -1:
            _my_sequence==0
            time.sleep(0.1)
            continue
        
        elif response.sequence>=0:
            return response


############################ Define our factors ####################################### 
def RSIIndividual(data):
    t = 6                           # 需要6天的历史数据
    close = np.array(data)[:, 5]    # 股票N天的close数据
    delta = np.diff(close)     

    delta = pd.Series(delta)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=5).mean()                      # 上涨天数的指数移动平均
    roll_down1 = down.abs().ewm(span=5).mean()            # 下跌天数的指数移动平均
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    return -RSI1.iloc[-1]


def SixDayRS(data):
    '''
    六日收益率反转
    '''
    t = 6                           # 需要6天的历史数据
    close = np.array(data)[:, 5]    # 股票N天的close数据    
    return -(close[-1]/close[-t] - 1) 

def MOM(data):
    t = 6                           # 需要6天的历史数据
    close = np.array(data)[:, 5]    # 股票N天的close数据
    delta = np.diff(close)     

    delta = pd.Series(delta)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=5).mean()                      # 上涨天数的指数移动平均
    roll_down1 = down.abs().ewm(span=5).mean()            # 下跌天数的指数移动平均
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    six_day_rs = -(close[-1]/close[-t] - 1) 

    return (-RSI1.iloc[-1] + six_day_rs)/2

def Size(data):
    '''
    市值（收盘价近似代替）
    '''
    t = 6                                   # 需要6天的历史数据
    close = np.array(data)[:, 5]            # 股票N天的close数据
    size = np.log(close)         # 按行求均值
    size = pd.Series(size)
    size = size.rolling(window=5).mean()

    return -size.iloc[-1]


def VolumePct(data):
    '''
    成交量变化率
    '''
    t = 6                                      # 需要6天的历史数据
    volume = np.array(data)[:, 6]              # 股票N天的volume数据
    
    volume = pd.Series(volume)
    # v_pct = np.abs(volume.pct_change(1))
    v_pct = volume.pct_change(3)
    v_pct = v_pct.rolling(window=5).mean()
    return -v_pct.iloc[-1]


###############################   并行计算   ##############################
def each_task(_key, _array):
    factors = []
    factors.append(RSIIndividual(_array))
    factors.append(SixDayRS(_array))
    factors.append(Size(_array))
    factors.append(VolumePct(_array))
    return (_key, factors)

def mp_task(batch, data):
    res = []
    for _key in batch:
        res.append(each_task(_key, data[_key]))
    return res

def _batch(stock_list):
    num_of_stock = len(stock_list)
    batch_size = num_of_stock // os.cpu_count()
    for i in range(0, num_of_stock, batch_size):
        yield stock_list[i: i+batch_size]


stub,session_key = initialize()
print('connected to server...')

# question channel
channel2 = grpc.insecure_channel('47.100.97.93:40722')
stub2 = question_pb2_grpc.QuestionStub(channel2)

column_names = ['Date','Instrument','Open','High','Low','Close','Volume','Amount']
data = {}

# 请求当前服务器最新的数据
my_sequence = 0
response = get_data(my_sequence, stub2)
# response = try_to_save_time(my_sequence, stub2)
my_sequence = response.sequence
print('first sequence:', my_sequence)
not_end = response.has_next_question

daily_data = np.array([stock.values for stock in response.dailystk])
daily_data = daily_data[:, 0:8]
num_of_stock = daily_data.shape[0]
stock_list = np.arange(num_of_stock)

print(daily_data.shape)

all_factors = [[]] * num_of_stock     # 每个股票的所有因子是一行

# 每一支股票单独存在字典里
for i in range(num_of_stock):
    data[i] = []
    data[i].append(daily_data[i])

# print(data)

# 因子或是回归最多利用过去10天的数据
MAX_NUM_RECORD = 10
num_of_holding = 100
leverage = 1

stock_batch = list(_batch(stock_list))
# print(stock_batch)

mp.set_start_method('fork')

while(True):
    # my_sequence += 1
    # print('my_sequence', my_sequence)
    # stub, session_key = initialize()
    # stub2 = question_pb2_grpc.QuestionStub(channel2)
    my_sequence += 1
    response = try_to_save_time(my_sequence, stub2)
    # response = get_data(my_sequence+1, stub2)
    # if response.sequence == -1:
    #     time.sleep(0.2)
    #     continue

    start =  time.time()
    not_end = response.has_next_question
    # my_sequence = response.sequence
    cur_capital = response.capital
    print('current day: {}, current capital: {}......\n'.format(my_sequence, cur_capital))

    position = [0]*num_of_stock
    send_positions(position ,stub, session_key, my_sequence)
    
    use_time = time.time() - start
    print("Time of posting position: %s", use_time)
    time.sleep(max(4-use_time, 0.1))