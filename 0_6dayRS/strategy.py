import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
import sys
warnings.filterwarnings("ignore")


data = pd.read_csv('../../contest-4/CONTEST_DATA_TEST_100_1.csv', usecols=[0,1,2,3,4,5,6],names=['date','stock','open','high','low','close','volume'])
print(data.head(3))

stocklist = list(set(data['stock']))
dic = {}
for i in stocklist:
    
    # 在这里计算想要用到的指标
    j = data[data['stock']==i]
    j.index = j['date']
    j['zhangfu'] = j['close'].diff()/j['close'].shift()
    j['100dayRS'] = j['close'].diff(100)/j['close'].shift(100)
    j['h/l'] = j['high']/j['low']
    j['cha']  = (j['close']-j['open']).abs()
    j['6'] = j['open']/j['close'].shift()
    j['20meanvolume'] = j['volume'].rolling(20).mean()
    j['6dayRS'] = j['close'].diff(6)/j['close'].shift(6)
    j['alpha118'] = (j['high']-j['open']).rolling(20).sum() / (j['open']-j['low']).rolling(20).sum()
    j['lottery'] = j['zhangfu'].rolling(20).max()


    # j.to_csv('./data/'+str(i)+'.csv')
    dic[i] = j


_all = []
sdf = [1]

shifenwei = {}
for iter in range(0,10):
    shifenwei[iter] = [1]


def positions():

    for i in range(25,len(list(set(data['date'])))-1):
        '''
        根据某一个因子初始化position
        '''
        positions = [0]*len(stocklist)
        for j in stocklist:
            #1、positions[j-1000] = dic[j].loc[i-2]['100dayRS']
            #2、positions[j-1000] = -dic[j].loc[i-2]['close']
            #3、positions[j-1000]  = -(np.corrcoef(dic[j].loc[i-10:i-2]['close'],dic[j].loc[i-10:i-2]['volume']))[0][1]
            #4、positions[j-1000]  = (np.std(dic[j].loc[i-6:i-2]['cha'])) 第三组最有效 所谓的std
            #5、positions[j-1000]  = (dic[j].loc[i-2]['6'])
            #6 positions[j-1000]  = -(dic[j].loc[i-2]['volume']/dic[j].loc[i-2]['20meanvolume'])
            # 7 
            # positions[j-1000] = -(dic[j].loc[i-2]['6dayRS'])
            # positions[j-1000]  = (np.std(dic[j].loc[i-21:i-2]['volume']))
            # 在这里引用想用的指标
            # positions[j-1000] = dic[j].loc[i-2]['alpha118']
            # 8
            positions[j-6000] = dic[j].loc[i-2]['6dayRS']
        
        positions = list((pd.Series(positions)).fillna(0))

        _pos = list(pd.Series(positions).rank())  # 对因子排序并调整建仓
        num_of_holding = 100                      # 同时持有股票数量
        for k in range(0,len(_pos)):
            if _pos[k]>=(500-num_of_holding/2):
                _pos[k] = -1
            elif _pos[k]<(num_of_holding/2):
                _pos[k]= 1
            else:
                _pos[k] = 0
        
        '''
        t确定仓位，t+1成交，t+2确认收益
        '''
        allsum = 0
        for j in range(0,len(_pos)):
            allsum+= dic[j+6000].loc[i]['zhangfu']*_pos[j]
        
        allsum = allsum/num_of_holding
        _all.append(allsum)
        sdf.append(sdf[-1]*(1+allsum))


        if(i%100 == 0):
            print(i)
    
    return _all, sdf

    


def pp_positions():
    
    for i in range(25,len(list(set(a['day'])))-1):
        '''
        根据某一个因子初始化position
        '''

        positions = [0]*len(stocklist)
        for j in stocklist:

            #1、positions[j-1000] = dic[j].loc[i-2]['100dayRS']
            #2、positions[j-1000] = -dic[j].loc[i-2]['close']
            #3、positions[j-1000]  = -(np.corrcoef(dic[j].loc[i-10:i-2]['close'],dic[j].loc[i-10:i-2]['volume']))[0][1]
            #4、positions[j-1000]  = (np.std(dic[j].loc[i-6:i-2]['cha'])) 第三组最有效 所谓的std
            #5、positions[j-1000]  = (dic[j].loc[i-2]['6'])
            #6 positions[j-1000]  = -(dic[j].loc[i-2]['volume']/dic[j].loc[i-2]['20meanvolume'])
            # 7 
            # positions[j-1000] = -(dic[j].loc[i-2]['6dayRS'])
            # positions[j-1000]  = (np.std(dic[j].loc[i-21:i-2]['volume']))
            # 在这里引用想用的指标
            # positions[j-1000] = dic[j].loc[i-2]['alpha118']
            # 8
            positions[j-1000] = dic[j].loc[i-2]['lottery']

        positions = list((pd.Series(positions)).fillna(0))

        # 画十分位图的话请把下面注释去掉，但是速度会明显变慢
        for iter in [0, 3, 6, 9]:
            _allsum = 0
            pp_pos = list(pd.Series(positions).rank())

            for k in range(0,len(pp_pos)):
                if pp_pos[k]>35*iter and pp_pos[k]<=35*(iter+1) :
                    pp_pos[k] = 1
                else:
                    pp_pos[k] = 0
                
            for j in range(0,len(pp_pos)):
                _allsum+= dic[j+1000].loc[i]['zhangfu']*pp_pos[j]
            
            _allsum = _allsum/35
            shifenwei[iter].append(shifenwei[iter][-1]*(1+_allsum))
        
        if(i%50 == 0):
            print(i)
        
    return shifenwei


if __name__ == '__main__':

    # print(len(sys.argv))

    if(sys.argv[1] == "1"):
        _all, sdf = positions()
        plt.plot(_all)
        plt.savefig('6dayRS_1_10.png')  # 每一天收益率的图
        plt.cla()

        plt.plot(sdf)
        plt.savefig('6dayRS_2_10.png')  # 做多前1/10,做空前1/10的绝对收益
        plt.cla()

    elif(sys.argv[1] == "2"):
        shifenwei = pp_positions()
        for iter in [0, 3, 6, 9]:
            plt.plot(shifenwei[iter],label=iter)
        plt.legend()
        # plt.savefig('ROC8_3.png')
        plt.cla() # 每个10分位的收益率
    
    else:
        print("选择测算方法：\n1-多空各1/10\n2-十分位做多\n")

    # print(a)