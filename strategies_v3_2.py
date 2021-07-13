'''
v_3_1

四因子 + Alpha101，去除相关性>0.3的，剩余8个因子
因子等权重组合

'''

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

import scipy.optimize as sco
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed



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
            time.sleep(0.1)
            continue
        
        elif response.sequence>=0:
            return response


############################ Define our factors ####################################### 

def SixDayRS(data):
    '''
    六日收益率反转
    '''
    t = 6                           # 需要6天的历史数据
    close = data[:, 5]              # 股票N天的close数据    
    # return (-(close[-1]/close[-t] - 1), _index)
    return -(close[-1]/close[-t] - 1)

###############################   Alpha 101   ##############################
def Alpha6(data):
    _open = pd.Series(data[:, 2])                  # 开盘价
    volume = pd.Series(data[:, 6])                 # 股票N天的volume数据
    alpha6 = _open.rolling(window=10).corr(volume) # 需要10天的历史数据
    return -1*alpha6.iloc[-1]


def Alpha9(data):
    close = pd.Series(data[:, 5])
    delta = close.diff(1)
    delta_min = delta.rolling(window=5).min().iloc[-1]     # 需要6天的历史数据
    if delta_min>0:
        return delta.iloc[-1]
    else:
        delta_max = delta.rolling(window=5).max().iloc[-1]
        if delta_max<0:
            return delta.iloc[-1]
        else:
            return -1*delta.iloc[-1]

def Alpha12(data):
    close = data[:, 5]
    amount = data[:, 7]                     # 股票2天的amount数据
    _sign = np.sign(amount[-1]-amount[-2])
    return -1*_sign*(close[-1]-close[-2])

def Alpha23(data):
    high = pd.Series(data[:, 3])
    _mean = high.rolling(10).mean()         # 需要10天数据
    _delta = high.diff(2)
    if _mean.iloc[-1]<high.iloc[-1]:
        return -1*_delta.iloc[-1]
    else:
        return 0

def Alpha51(data):
    close = data[:, 5]                      # 需要20天数据
    condition1 = (close[-20]-close[-10])/10 - (close[-10]-close[-1])/10
    if condition1 < -0.05:
        return 1
    else:
        return -1*(close[-1]-close[-2])

def Alpha53(data):
    high = data[:, 3]                       # 需要10天数据
    low = data[:, 4]
    close = data[:, 5]
    x1 = ((close[-1]-low[-1])-(high[-1]-close[-1]))/(close[-1]-low[-1])
    x9 = ((close[-9]-low[-9])-(high[-9]-close[-9]))/(close[-9]-low[-9])

    return -1*(x1-x9)

def Alpha54(data):
    _open = data[:, 2]                      # 需要2天数据
    _high = data[:, 3]
    _low = data[:, 4]
    _close = data[:, 5]
    fenzi = -1*(_low[-1]-_close[-1])*(_open[-1]**5)
    fenmu = (_low[-1]-_high[-1])*(_close[-1]**5)
    return fenzi/fenmu

###############################   并行计算因子   ##############################
def each_task(_key, _array):
    factors = []
    _array = np.array(_array)

    factors.append(Alpha12(_array))
    factors.append(Alpha54(_array))

    factors.append(Alpha9(_array))
    factors.append(SixDayRS(_array)) 

    factors.append(Alpha6(_array))
    factors.append(Alpha23(_array))
    factors.append(Alpha53(_array))

    factors.append(Alpha51(_array))
    
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



if __name__ == "__main__":
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

    # 每一支股票单独存在字典里
    for i in range(num_of_stock):
        data[i] = []
        data[i].append(daily_data[i])

    # print(data)

    # 因子或是回归最多利用过去20天的数据
    MAX_NUM_RECORD = 20
    num_of_holding = 100
    leverage = 1.0

    yesterday_close = []  # 记录昨日收盘价，用于计算当日收益率
    pct_record = []         # 记录每日收益率
    factors_record = []     # 记录每日因子值
    factors_weight = []     # 因子加权权重

    stock_batch = list(_batch(stock_list))
    # print(stock_batch)

    mp.set_start_method('fork')

    while(True):
        all_factors = [[]] * num_of_stock     # 每个股票的所有因子是一行
        # my_sequence += 1
        # print('my_sequence', my_sequence)
        # stub, session_key = initialize()
        try:
            stub2 = question_pb2_grpc.QuestionStub(channel2)
            response = get_data(my_sequence+1, stub2)
        except:
            time.sleep(0.1)
            continue
        if response.sequence == -1:
            time.sleep(0.1)
            continue

        start =  time.time()
        not_end = response.has_next_question
        my_sequence = response.sequence
        cur_capital = response.capital
        print('current day: {}, current capital: {}......\n'.format(my_sequence, cur_capital))

        daily_data = np.array([stock.values for stock in response.dailystk])
        daily_data = daily_data[:, 0:8]
        today_close = daily_data[:, 5]                    # 当日收盘价
        if len(yesterday_close)>0:                        # 从第二天开始算起
            pct = (today_close - yesterday_close)/yesterday_close
            pct_record.append(pct)

        num_of_stock = daily_data.shape[0]
        stock_list = np.arange(num_of_stock)

        print(daily_data.shape)

        # 每一支股票单独存在字典里
        for i in range(num_of_stock):
            data[i].append(daily_data[i])

        if len(data[0]) >= 20:

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(mp_task, batch, data) for batch in stock_batch]
                for future in as_completed(futures):
                    factors_batch = future.result()
                    for i in range(len(factors_batch)):
                        _key, factors = factors_batch[i]
                        # print(_key, factors)
                        all_factors[_key] = factors

            # print(all_factors)
            print(len(all_factors))                                               # N * K

            all_factors = np.array(all_factors)
            all_factors[np.isnan(all_factors)] = 0                                # 处理空值
            all_factors[np.isinf(all_factors)] = 0                                # 处理无限值
            all_factors = preprocessing.scale(all_factors)                        # 按列z-score标准化
            # all_factors = np.hstack(([[1]]*num_of_stock, all_factors))            # 每一行最左侧插入[1]
            factors_record.append(all_factors)                                    # T * N * (K)，储存一段时间的因子值，用于计算因子收益率
            num_of_factor = all_factors.shape[1]
            
            if len(factors_weight) == 0:
                final_factors = np.array(all_factors).mean(axis=1)
            else:
                final_factors = all_factors @ factors_weight
            
            _rank = np.argsort(np.argsort(final_factors))    # 0-499，多因子等权组合之后的排序值
            _pos = np.array([0]*num_of_stock)
            
            _buy_line = num_of_stock-num_of_holding/2
            _sell_line = num_of_holding/2
            _pos[_rank>=_buy_line] = 1
            _pos[_rank<_sell_line] = -1
            
            position = _pos * cur_capital * leverage / num_of_holding / today_close
            position[np.isnan(position)] = 0
            position = list(position.astype(int))
            print("len of position: ", len(position))
            # print(position)
            
            try:
                stub, session_key = initialize()
                send_positions(position ,stub, session_key, my_sequence)
            except:
                time.sleep(0.1)
                stub, session_key = initialize()
                send_positions(position ,stub, session_key, my_sequence)
                continue
            
            factors_record_len = len(factors_record)
            if factors_record_len >= 25:
                factors_weight = []
                for i in range(factors_record_len-2):
                    factors_tmp = factors_record[-(i+3)]                    # N * K
                    pct_tmp = pct_record[-(i+1)]                            # N * 1
                    factors_corr = np.cov(factors_tmp, rowvar=0)
                    factors_pct_corr = [np.cov(factors_tmp[:, j], pct_tmp)[0,1] for j in range(num_of_factor)]
                    factors_weight.append(factors_corr @ factors_pct_corr)
                factors_weight = np.array(factors_weight).mean(axis=0)      # 按列平均
                factors_weight /= np.sum(factors_weight)

                # print("factors_weight: ", factors_weight)
                if factors_record_len >= 25: factors_record.pop(0)

                use_time = time.time() - start
                print("Time of 🍊: ", round(use_time,2))

            # 删除最老的一天的数据
            if len(data[0]) >= 22:
                for i in range(num_of_stock):
                    data[i].pop(0)
            
        use_time = time.time() - start
        print("Time of posting position: %s", use_time)
        time.sleep(max(4.5-use_time, 0.1))
        
        yesterday_close = today_close