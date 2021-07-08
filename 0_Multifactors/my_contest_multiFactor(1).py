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

import grpc

import question_pb2
import question_pb2_grpc

# --------

from concurrent import futures


import contest_pb2
import contest_pb2_grpc

import time
import re

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



def initialize():

    channel = grpc.insecure_channel('47.103.23.116:56702')
    stub = contest_pb2_grpc.ContestStub(channel)
    response = stub.login(contest_pb2.LoginRequest(user_id = 69, user_pin ='vKfPGNrn'))
    print(response)
    return stub,response.session_key


def get_data(_sequence,stub2):

    response2 = stub2.get_question(question_pb2.QuestionRequest(user_id = 69, sequence = _sequence))
    return response2

def send_positions(_positions,_stub,_session_key,_sequence):
    response3 = stub.submit_answer(contest_pb2.AnswerRequest(   user_id = 69, \
                                                                user_pin ='vKfPGNrn', \
                                                                session_key = _session_key, \
                                                                sequence = _sequence,\
                                                                positions = _positions ))
    print(response3)


def try_to_save_time(_my_sequence, _stub2):
    while True:
        response = get_data(_my_sequence,_stub2)
        _new_sequence = response.sequence
        
        if _new_sequence != _my_sequence:
            return _new_sequence

        time.sleep(0.2)


def GetIndex(name):
    # ['day','stock','open','high','low','close','volume']

    if name=='day':
        return 0
    elif name=='stock':
        return 1
    elif name=='open':
        return 2
    elif name=='high':
        return 3
    elif name=='low':
        return 4
    elif name=='close':
        return 5
    elif name=='volume':
        return 6
        
def Calculate_ZhangFu(AllData, Days, length):
    if Days > 1:
        latestClose = AllData[-1][:, 5]
        beforeLatesClose = AllData[-2][:, 5]
        return (latestClose-beforeLatesClose)/beforeLatesClose
    else:
        return [0]*length

def Calculate_SixROI(AllData, Days, length):
    if Days > 7:
        latestClose = AllData[-1][:, 5]
        beforeLatesClose = AllData[-7][:, 5]
        return list(-(latestClose-beforeLatesClose)/beforeLatesClose)
    else:
        return [0]*length


def Calculate_Close(AllData, Days, length):
    if Days > 7:
        latestClose = AllData[-1][:, 5]
        return list(-latestClose)
    else:
        return [0]*length

def Calculate_TwentyMax(AllData, Days, length):
    if Days > 19:
        AllData = np.array(AllData)
        last20Close = AllData[:, :, 5]
        return list(last20Close[-20:, :].max(0))
    else:
        return [0]*length

def Calculate_Alpha118(AllData, Days, length):
    if Days > 19:
        AllData = np.array(AllData)
        last20High = AllData[:, :, 3]
        last20Open = AllData[:, :, 2]
        last20Low = AllData[:, :, 4]

        sumOfHigh = (last20High-last20Open)[-20:, :].sum(0)
        sumOfLow = (last20Open-last20Low)[-20:, :].sum(0)

        return list(sumOfHigh/sumOfLow)
    else:
        return [0]*length

# 一些初始化

stub,session_key = initialize()
channel2 = grpc.insecure_channel('47.103.23.116:56701')
stub2 = question_pb2_grpc.QuestionStub(channel2)
my_sequence = 0

dic = {}

my_sequence = try_to_save_time(my_sequence,stub2)

# 远程主机返回的所有股票数据都储存起来
# AllData将是一个储存二维数组的list
AllData = []

# 储存我们已经实际接收到的数据的天数
Days = 0

## 正式运行
print("I'm going...")
while(True):
    
    start =  time.time()
    response = get_data(my_sequence,stub2)
    # my_sequence = response.sequence
    Days += 1

    print(response.has_next_question,response.capital,response.sequence)
    """ 
    每天远程主机返回的数据都储存为一个二维数组
    每一行：代表一支股票
    每一列：['day','stock','open','high','low','close','volume']
    """
    NumOfStock = len(response.dailystk)
    OneDayData = np.array([response.dailystk[i].values for i in range(0, NumOfStock)])
    stocklist = OneDayData[:, 1]
    lenOfStock = len(stocklist)

    AllData.append(OneDayData)
    
    # 计算不同因子
    # ZhangFu_li = Calculate_ZhangFu(AllData, Days, lenOfStock)
    SixROI_li = Calculate_SixROI(AllData, Days, lenOfStock)
    # TwentyMax_li = Calculate_TwentyMax(AllData, Days, lenOfStock)
    # Alpha118_li = Calculate_Alpha118(AllData, Days, lenOfStock)
    Close_li = Calculate_Close(AllData, Days, lenOfStock)



    use_time = time.time() - start
    print("Time of processing data: ", use_time)

    # 以下就可以开始写策略了。
    # 两个因子
    positions_1 = (pd.Series(SixROI_li)).fillna(0)
    # positions_1 = positions_1.rank()
    # positions_1 = positions_1.fillna(0)

    positions_2 = (pd.Series(Close_li)).fillna(0)
    # positions_2 = positions_2.rank()
    # positions_2 = positions_2.fillna(0)
    
    # 计算因子组合，可以设定不同的权重
    # _pos = list((1*positions_1 + 1*positions_2).rank().fillna(0))
    _pos = list((positions_1).rank().fillna(0))
    _pos2 = list((positions_2).rank().fillna(0))
    _finale_position = [0]*351

    for k in range(0,len(_pos)):
        if _pos[k]>314:
            _finale_position[k] += 0.5
        elif _pos[k] <=35:
            _finale_position[k] -= 0.5
        # else:
        #     _pos[k] = 0
    for k in range(0,len(_pos2)):
        if _pos2[k]>333:
            _finale_position[k] += 1
        elif _pos2[k] <=18:
            _finale_position[k] -= 1

    moneyspent = 0

    # for j in stocklist:
    #     moneyspent= abs(dic[j].loc[current_days]['close']* _pos[int(j-1000)])+moneyspent
    # print(moneyspent)
    
    # _pos = [i * response.capital*1.95/moneyspent for i in _pos]
    for k in range(0,len(_pos)):
        _finale_position[k] = _finale_position[k] * response.capital/70/AllData[-1][k, 5]

    _pos = list((pd.Series(_pos)).fillna(0))

    use_time = time.time() - start
    print("Time of caculating position: ", use_time)

    # for k in range(0,len(_pos)):
    #     _pos[k] = _pos[k]/dic[k+1000].loc[current_days]['close']

    # 以下不用管，是框架，以上是策略。

    send_positions(_finale_position ,stub,session_key,my_sequence)

    use_time = time.time() - start
    print("Time of posting position: %s", use_time)
    time.sleep(max(2-use_time,0.1))


    my_sequence = try_to_save_time(my_sequence,stub2)



    
   
