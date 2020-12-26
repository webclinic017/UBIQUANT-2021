import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# import warnings
import sys
# warnings.filterwarnings("ignore")


a = pd.read_csv('CONTEST_DATA_IN_SAMPLE_2.csv',names=['day','stock','open','high','low','close','volume'])

FactorNum = 9

stocklist = list(set(a['stock']))
dic = {}
for i in stocklist:
    
    # 在这里计算想要用到的指标(FactorNum += 1)
    j = a[a['stock']==i]
    j.index = j['day']
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

print("number of testing factors: ", FactorNum)


def Positions(index):
    _all = []
    sdf = [1]

    for i in range(25,len(list(set(a['day'])))-1):
        '''
        根据某一个因子初始化position
        '''
        positions = [0]*len(stocklist)
        for j in stocklist:
            #1、positions[j-1000] = dic[j].loc[i-2]['100dayRS']
            if index==1:
                positions[j-1000] = dic[j].loc[i-2]['100dayRS']
            
            #2、positions[j-1000] = -dic[j].loc[i-2]['close']
            elif index==2:
                positions[j-1000] = -dic[j].loc[i-2]['close']

            #3、positions[j-1000]  = -(np.corrcoef(dic[j].loc[i-10:i-2]['close'],dic[j].loc[i-10:i-2]['volume']))[0][1]
            elif index==3:
                positions[j-1000]  = -(np.corrcoef(dic[j].loc[i-10:i-2]['close'],dic[j].loc[i-10:i-2]['volume']))[0][1]

            #4、positions[j-1000]  = (np.std(dic[j].loc[i-6:i-2]['cha'])) 第三组最有效 所谓的std
            elif index==4:
                positions[j-1000]  = (np.std(dic[j].loc[i-6:i-2]['cha']))

            #5、positions[j-1000]  = (dic[j].loc[i-2]['6'])
            elif index==5:
                positions[j-1000]  = (dic[j].loc[i-2]['6'])

            #6 positions[j-1000]  = -(dic[j].loc[i-2]['volume']/dic[j].loc[i-2]['20meanvolume'])
            elif index==6:
                positions[j-1000]  = -(dic[j].loc[i-2]['volume']/dic[j].loc[i-2]['20meanvolume'])

            #7 positions[j-1000] = -(dic[j].loc[i-2]['6dayRS'])
            elif index==7:
                positions[j-1000] = -(dic[j].loc[i-2]['6dayRS'])
            
            # 在这里引用想用的指标
            #8 positions[j-1000] = dic[j].loc[i-2]['alpha118']
            elif index==8:
                positions[j-1000] = dic[j].loc[i-2]['alpha118']
            
            #9 positions[j-1000] = dic[j].loc[i-2]['lottery']
            elif index==9:
                positions[j-1000] = dic[j].loc[i-2]['lottery']
        
        positions = list((pd.Series(positions)).fillna(0))

        '''
        对因子排序并调整建仓
        '''
        _pos = list(pd.Series(positions).rank())
        for k in range(0,len(_pos)):
            if _pos[k]>316:
                _pos[k] = 1
            elif _pos[k]<=35:
                _pos[k]= -1
            else:
                _pos[k] = 0
        
        '''
        t确定仓位，t+1成交，t+2确认收益
        '''
        allsum = 0
        for j in range(0,len(_pos)):
            allsum+= dic[j+1000].loc[i]['zhangfu']*_pos[j]
        
        allsum = allsum/70
        _all.append(allsum)
        sdf.append(sdf[-1]*(1+allsum))


        if(i%100 == 0):
            print(i)
    
    return _all, sdf

    


def PP_Positions():
    
    shifenwei = {}
    for iter in range(0,10):
        shifenwei[iter] = [1]
    
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

def Correlaion(dic):
    all_list = []
    sdf_list = []
    #for i in range(1, FactorNum+1):
    for i in [6, 7, 8, 9]:
        print("calculating factor {} ......".format(i))
        _all, sdf = Positions(i)
        all_list.append(_all)
        sdf_list.append(sdf)
        
    return all_list, sdf_list

if __name__ == '__main__':

    # print(len(sys.argv))

    if(sys.argv[1] == "1"):
        _all, sdf = Positions(9)
        plt.plot(_all)
        plt.savefig('../result/lottery_1_10.png')  # 每一天收益率的图
        plt.cla()

        plt.plot(sdf)
        plt.savefig('../result/lottery_2_10.png')  # 做多前1/10,做空前1/10的绝对收益
        plt.cla()

    elif(sys.argv[1] == "2"):
        shifenwei = PP_Positions()
        for iter in [0, 3, 6, 9]:
            plt.plot(shifenwei[iter],label=iter)
        plt.legend()
        plt.savefig('../result/ROC8_3.png')
        plt.cla() # 每个10分位的收益率
    
    elif(sys.argv[1] == "3"):
        all_list, sdf_list = Correlaion(dic)
        

        all_column = ['100dayRS', 'close', 'volume', 'cha', '6', '20meanvolume',
                    '6dayRS', 'alpha118', 'lottery']
        datadic = {}
        column = all_column[5:]
        for col, factor in zip(column, all_list):
            datadic[col] = factor
        
        factor_df = pd.DataFrame(datadic)
        factor_df.to_csv('ROI.csv')
        cor1 = factor_df.corr()
        print(cor1)

    else:
        print("选择测算方法：\n1-多空各1/10\n2-十分位做多\n")

    # print(a)