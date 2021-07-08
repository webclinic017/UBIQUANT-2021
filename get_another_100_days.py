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
from mfm.MFM import MFM

import warnings
warnings.filterwarnings("ignore")

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
while True:
    try:
        stub,session_key = initialize()
        break
    except:
        time.sleep(1)

print('connected to server...')

# question channel
channel2 = grpc.insecure_channel('47.100.97.93:40722')
stub2 = question_pb2_grpc.QuestionStub(channel2)

column_names = ['Date', 'Instrument', 'Open', 'High', 'Low', 'Close', 'Volumn', 'Amount']
alpha_name = ['Alpha_']*100
for i in range(1, 101, 1):
    alpha_name = 'Alpha_' + str(i)
    column_names.append(alpha_name)

# column_names = ['Date','Instrument','Open','High','Low','Close','Volume','Amount']
data = pd.DataFrame(columns=column_names)

# 请求当前服务器最新的数据
my_sequence = 0
response = try_to_save_time(my_sequence, stub2)
my_sequence = response.sequence

num_of_dates = 1

# 全局参数
num_of_holding = 100   # 总共持有的股票数量
leverage = 1
not_end = True

while(not_end):
    my_sequence += 1
    response = try_to_save_time(my_sequence,stub2)
    # print(np.array(response.dailystk)[0])
    start =  time.time()
    not_end = response.has_next_question
    cur_day = response.sequence
    cur_capital = response.capital
    print("\n===========================")
    print('current day: {}, current capital: {}......\n'.format(cur_day, cur_capital))

    if cur_day < 890:
        continue
    daily_data = np.array(response.dailystk)
    daily_data = np.array([stock.values for stock in daily_data])
 
    daily_data = pd.DataFrame(daily_data, columns=column_names)
    daily_data[['Date', 'Instrument']] = daily_data[['Date', 'Instrument']].astype(int)
    num_of_stock = daily_data.shape[0]

    data = data.append(daily_data, ignore_index=True)
    num_of_dates += 1
    # print(data)
    
    print("shape of data:", data.shape)

    if cur_day == 98:
        data.to_csv('another_100_data.csv')

data.to_csv('another_100_data.csv')



    

    
   
