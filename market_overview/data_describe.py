# %%
import pandas as pd


# %%
data = pd.read_csv(r"../../contest-4/CONTEST_DATA_TEST_100_1.csv",usecols=[0,1,2,3,4,5,6],names=['Open time','Instrument','Open','High','Low','Close','Volume'])
print(data.shape)
print("")
print(data.head(3))
print("")

num_of_stocks = len(pd.unique(data['Instrument']))
print("number of stocks: {}".format(num_of_stocks))

num_of_dates = len(pd.unique(data['Open time']))
print("number of dates: {}".format(num_of_dates))