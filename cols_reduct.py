import pandas as pd
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# get all categorical columns

categ_train_df = train_df.select_dtypes(include='O')
categ_test_df = test_df.select_dtypes(include='O')

# Dropping uneeded categorical Features Columns 
sample_train_df = categ_train_df.drop(['PoolQC', 'Fence', 'MiscFeature', 'Alley'], axis=1)
sample_test_df = categ_test_df.drop(['PoolQC', 'Fence', 'MiscFeature', 'Alley'], axis=1)

# Dropping numerical Features columns with corr < 0.5 with numerical Features
num_train_df = train_df[['OverallQual','YearBuilt','FullBath','GrLivArea','TotalBsmtSF','1stFlrSF', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']]
num_test_df = train_df[['OverallQual','YearBuilt','FullBath','GrLivArea','TotalBsmtSF','1stFlrSF', 'TotRmsAbvGrd',  'GarageCars', 'GarageArea']]


result_train_df = pd.concat([num_train_df, sample_train_df], axis=1)
result_test_df = pd.concat([num_test_df, sample_test_df], axis=1)

# final dataframe
train_df = result_train_df
test_df = result_test_df


