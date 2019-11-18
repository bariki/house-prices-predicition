from imputation1 import df_train,pd,np

df_correlated = df_train.corr()

ser = (df_correlated.loc['SalePrice']).sort_values(ascending=False).head(20)
ser = ser[ser > 0.51]
ser = ser.drop('SalePrice')

num_features = np.array(ser.index)