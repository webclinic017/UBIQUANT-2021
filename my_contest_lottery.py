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

def try_to_save_time(_day, _my_sequence, _stub2):
    while True:
        response = get_data(_my_sequence,_stub2)
        _my_sequence = response.sequence

        newKey = str(response.dailystk[1]).split('\n')
        newKey = [float(j[8:]) for j in newKey if j != '']
        
        if _day!=newKey[0]:
            return _my_sequence

        time.sleep(0.05)

    

# 一些初始化

stub,session_key = initialize()
channel2 = grpc.insecure_channel('47.103.23.116:56701')
stub2 = question_pb2_grpc.QuestionStub(channel2)
my_sequence = 0

dic = {}

response = get_data(my_sequence,stub2)
my_sequence = response.sequence
for i in range(0,len(response.dailystk)):
    p1 = r"values: " 
    newKey = (re.sub(p1, "", str(response.dailystk[i]))).split('\n')
    newKey = [float(i) for i in newKey if i != '']
    dic[newKey[1]] = pd.DataFrame(columns = ['open','high','low','close','volume'])

all_dic = pd.DataFrame(columns = ['day','stock','open','high','low','close','volume'])

## 初始请求时间计算，为了尽量能够更早的

response = get_data(my_sequence,stub2)
my_sequence = response.sequence


current_days = my_sequence
my_sequence = try_to_save_time(current_days,my_sequence,stub2)


## 正式运行
while(True):
    
    start =  time.time()
    response = get_data(my_sequence,stub2)
    my_sequence = response.sequence

    print(response.has_next_question,response.capital,response.sequence,response.positions)

    current_days = 0
    print(current_days)
    temp_dic = pd.DataFrame(columns = ['day','stock','open','high','low','close','volume'])


    for i in range(0,len(response.dailystk)):
        # newKey = str(response.dailystk[i]).split('\n')
        # newKey = [float(j[8:]) for j in newKey if j != '']
        
        # temp_dic = temp_dic.append(pd.Series(response.dailystk[i].values),ignore_index=True)
        temp_dic = temp_dic.append({'day':response.dailystk[i].values[0],'stock':int(response.dailystk[i].values[1]), 'open':response.dailystk[i].values[2],'high':response.dailystk[i].values[3],'low':response.dailystk[i].values[4],'close':response.dailystk[i].values[5],'volume':response.dailystk[i].values[6]}, ignore_index=True)

    all_dic = all_dic.append(temp_dic) # 这样更节省时间
     

    # 获取当前天数，为了能够最大限度在下一天刚一转换时搞定
    current_days = response.sequence

    
    # 把全部数据分拆到股票各自的列表里。需要什么指标的话在这里直接算，节省时间。
    stocklist = list(set(all_dic['stock']))
    dic = {}
    for i in stocklist:
        j = all_dic[all_dic['stock']==i]
        j.index = j['day']
        j['zhangfu'] = j['close'].diff()/j['close'].shift() # 类似这样的指标什么的
        j['6dayRS'] = j['close'].diff(6)/j['close'].shift(6)
        j['lottery'] = j['zhangfu'].rolling(20).max()
        dic[i] = j


    # 以下就可以开始写策略了。

    positions = [0]*len(stocklist)
    for j in stocklist:
        # positions[int(j-1000)] = dic[j].loc[current_days]['100dayRS']
        # positions[int(j-1000)] = -dic[j].loc[current_days]['close']
        positions[int(j-1000)] = -dic[j].loc[current_days]['lottery']


    positions = list((pd.Series(positions)).fillna(0))
    _pos = list(pd.Series(positions).rank())
    _pos = list((pd.Series(_pos)).fillna(0))

    for k in range(0,len(_pos)):
        if _pos[k]>316:
            _pos[k] = 1
        elif _pos[k]<=35:
            _pos[k]= -1
        else:
            _pos[k] = 0

    moneyspent = 0

    # for j in stocklist:
    #     moneyspent= abs(dic[j].loc[current_days]['close']* _pos[int(j-1000)])+moneyspent
    # print(moneyspent)
    
    # _pos = [i * response.capital*1.95/moneyspent for i in _pos]
    for k in range(0,len(_pos)):
        _pos[k] = _pos[k] * response.capital*1.95/140/dic[k+1000].loc[current_days]['close']

    _pos = list((pd.Series(_pos)).fillna(0))


    # for k in range(0,len(_pos)):
    #     _pos[k] = _pos[k]/dic[k+1000].loc[current_days]['close']


    # 以下不用管，是框架，以上是策略。

    send_positions(_pos ,stub,session_key,my_sequence)

    use_time = time.time() - start
    print(use_time)
    time.sleep(max(4.75-use_time,0.1))

    my_sequence = try_to_save_time(current_days,my_sequence,stub2)

    
   
