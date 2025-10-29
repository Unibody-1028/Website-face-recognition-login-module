# time: 2024/7/1 10:34
# creater:guopengpeng

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# 读取CSV文件
df = pd.read_csv('./data/features_total.csv')
#print(df.iloc[0,0])
#print(df.shape)
print(df.shape)
features = df.iloc[:, 1:]


distances = pdist(features, 'euclidean')

print(distances)

