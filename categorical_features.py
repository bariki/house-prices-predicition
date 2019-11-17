from imputation1 import df_train,pd,np

def changeQualityToInt(val):
    if(val == 'Ex'):
        return 5
    elif(val == 'Gd'):
        return 4
    elif(val == 'TA'):
        return 3
    elif(val == 'Fa'):
        return 2
    elif(val == 'Po'):
        return 1
    else:
        return 0


def changeGarageFinishToInt(val):
    if(val == 'Fin'):
        return 3
    elif(val == 'RFn'):
        return 2
    elif(val == 'Unf'):
        return 1
    else:
        return 0



df_train['BsmtCond'] = pd.DataFrame(list(map(changeQualityToInt, df_train['BsmtCond'])))
df_train['BsmtQual'] = pd.DataFrame(list(map(changeQualityToInt, df_train['BsmtQual'])))
df_train['PoolQC'] = pd.DataFrame(list(map(changeQualityToInt, df_train['PoolQC'])))
df_train['KitchenQual'] = pd.DataFrame(list(map(changeQualityToInt, df_train['KitchenQual'])))
df_train['HeatingQC'] = pd.DataFrame(list(map(changeQualityToInt, df_train['HeatingQC'])))
df_train['GarageQual'] = pd.DataFrame(list(map(changeQualityToInt, df_train['GarageQual'])))
df_train['GarageCond'] = pd.DataFrame(list(map(changeQualityToInt, df_train['GarageCond'])))
df_train['GarageFinish'] = pd.DataFrame(list(map(changeGarageFinishToInt, df_train['GarageFinish'])))
df_train['FireplaceQu'] = pd.DataFrame(list(map(changeQualityToInt, df_train['FireplaceQu'])))
df_train['ExterCond'] = pd.DataFrame(list(map(changeQualityToInt, df_train['ExterCond'])))
df_train['ExterQual'] = pd.DataFrame(list(map(changeQualityToInt, df_train['ExterQual'])))


res = df_train[['BsmtCond', 'BsmtQual', 'PoolQC' , 'KitchenQual', 'HeatingQC', 'GarageQual', 'GarageCond', 'GarageFinish', 'FireplaceQu', 'ExterCond', 'ExterQual', 'SalePrice']]

res = res.corr()

# take only with correlation > 0.5
res[res['SalePrice'] > 0.5].sort_values(by='SalePrice', ascending=False)