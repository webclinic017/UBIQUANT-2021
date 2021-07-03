import numpy as np
import pandas as pd

use_columns = [1, 5]                           # 股票id和收盘价
use_columns.extend(list(np.arange(8,108,1)))   # 100个因子
df = pd.read_csv(r"../CONTEST_DATA_TEST_100_1.csv", header=None, usecols=use_columns)

print(df.head())

result = df.iloc[:, 2:].corr()
result.to_csv(r'correlation.csv')

print(result)