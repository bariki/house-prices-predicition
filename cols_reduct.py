import pandas as pd
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Dropping Columns with categorical Features

sample_train_df = train_df.drop(['Id','PoolQC', 'Fence', 'MiscFeature', 'Alley'], axis=1)
sample_test_df = test_df.drop(['Id','PoolQC', 'Fence', 'MiscFeature', 'Alley'], axis=1)

# Dropping Columns with numerical Features

num_train_df = train_df[['OverallQual','YearBuilt','FullBath','GrLivArea','TotalBsmtSF','1stFlrSF', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']]
num_test_df = train_df[['OverallQual','YearBuilt','FullBath','GrLivArea','TotalBsmtSF','1stFlrSF', 'TotRmsAbvGrd',  'GarageCars', 'GarageArea']]

# combine the resulted dataframes
result_train_df = pd.concat([num_train_df, sample_train_df], axis=1)
result_test_df = pd.concat([num_test_df, sample_test_df], axis=1)

