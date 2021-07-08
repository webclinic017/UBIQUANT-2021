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
        
        if response.sequence == -1:
            _my_sequence==0
            time.sleep(0.05)
            continue
        
        elif response.sequence>=0:
            return response

        

# contest channel
stub,session_key = initialize()

# question channel
channel2 = grpc.insecure_channel('47.100.97.93:40722')
stub2 = question_pb2_grpc.QuestionStub(channel2)

# 请求当前服务器最新的数据
my_sequence = 0
response = try_to_save_time(my_sequence, stub2)
my_sequence = response.sequence

all_all_list = {}

## 正式运行
while(True):
    
    my_sequence += 1
    response = try_to_save_time(my_sequence,stub2)
    
    start =  time.time()
    not_end = response.has_next_question
    cur_day = response.sequence
    cur_capital = response.capital
    print("\n" + "======"*10)
    print('current day: {}, current capital: {}......\n'.format(cur_day, cur_capital))
    temp_dic = pd.DataFrame(columns = ['Date','Instrument','Open','High','Low','Close','Volume','Amount'])

    all_list = []

    day6maelist = None

    stock_numbers = len(response.dailystk)

    for i in range(0,stock_numbers):
        temp_dic = temp_dic.append({'day':response.dailystk[i].values[0],'stock':int(response.dailystk[i].values[1]), 'open':response.dailystk[i].values[2],'high':response.dailystk[i].values[3],'low':response.dailystk[i].values[4],'close':response.dailystk[i].values[5],'volume':response.dailystk[i].values[6]}, ignore_index=True)
        all_list.append(response.dailystk[i].values)

    try:
        day6maelist = all_all_list[response.sequence - 6]
    except:
        all_all_list[response.sequence] = all_list
        continue

    use_time = time.time() - start
    print("Time of processing data: %s", use_time)

    positions = [0]*stock_numbers
    # for j in stocklist:
    #     positions[int(j-1000)] = -dic[j].loc[current_days]['6dayRS']
    for j in range(0,stock_numbers):
        today_close = all_list[j][5]
        before6day_close = day6maelist[j][5]
        positions[j] = -(today_close-before6day_close) / before6day_close


    # all_dic = all_dic.append(temp_dic) # 这样更节省时间
     
    
    # 把全部数据分拆到股票各自的列表里。需要什么指标的话在这里直接算，节省时间。
    # stocklist = list(set(all_dic['stock']))
    # dic = {}
    # for i in stocklist:
    #     j = all_dic[all_dic['stock']==i]
    #     j.index = j['day']
    #     j['zhangfu'] = j['close'].diff()/j['close'].shift() # 类似这样的指标什么的
    #     j['6dayRS'] = j['close'].diff(6)/j['close'].shift(6)
    #     dic[i] = j

    # 以下就可以开始写策略了。


    positions = list((pd.Series(positions)).fillna(0))
    _pos = list(pd.Series(positions).rank())
    _pos = list((pd.Series(_pos)).fillna(0))

    for k in range(0,len(_pos)):
        if _pos[k]>465:
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
        # _pos[k] = _pos[k] * response.capital*2/140/dic[k+1000].loc[current_days]['close']
        _pos[k] = _pos[k] * response.capital*2/140/all_list[k][5]

    _pos = list((pd.Series(_pos)).fillna(0))
    
    use_time = time.time() - start
    print("Time of caculating position: %s", use_time)

    # for k in range(0,len(_pos)):
    #     _pos[k] = _pos[k]/dic[k+1000].loc[current_days]['close']


    # 以下不用管，是框架，以上是策略。
    print(_pos)
    print(len(_pos))
    send_positions(_pos ,stub,session_key,my_sequence)


    all_all_list[response.sequence] = all_list

    use_time = time.time() - start
    print("Time of posting position: %s", use_time)
    time.sleep(max(4.75-use_time,0.1))

    

    
   
