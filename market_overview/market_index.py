import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(r"../../contest-4/CONTEST_DATA_TEST_100_1.csv",usecols=[0,1,2,3,4,5,6],names=['Open time','Instrument','Open','High','Low','Close','Volume'])
data['Pct'] = data.groupby('Instrument')['Close'].pct_change()

market_index = data.groupby('Open time')['Pct'].mean()
market_index.iloc[0] = 0
market_index += 1
market_index = market_index.cumprod()
print(market_index)

plt.plot(market_index)
plt.savefig(r"market_index_equal.png")