import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

url = "https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/King_County/kingCountyHouseData.csv"
df = pd.read_csv(url)

columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors','price']
df = df.loc[:,columns]
# print(df.head(10))
# # [10 rows x 6 columns]

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors']
x= df.loc[:, features]      # image
y= df.loc[:, ['price']]     # label
# print(x)
# # [21613 rows x 5 columns]
# print(y)
# # [21613 rows x 1 columns]
# print(x.shape, y.shape)
# # (21613, 5) (21613, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = 0.8)
print(x_train.shape, x_test.shape)
# 0.75 (16209, 5) (5404, 5)  0.8 (17290, 5) (4323, 5)
print(y_train.shape, y_test.shape)
# 0.75 (16209, 1) (5404, 1)  0.8 (17290, 1) (4323, 1)
