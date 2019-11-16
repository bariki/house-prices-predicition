import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.model_selection import cross_val_score
import copy

df_train = pd.read_csv("train.csv")
df_test =  pd.read_csv("test.csv")

#preview dataframe
df_train.head(5)
df_test.head(5)

#checking shapes
df_train.shape
df_test.shape

#preverving houses ID
train_ID = df_train['Id']
test_ID = df_test['Id']

df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)


#We find out there is missing data, therefore lets combine there dataframe
#to big one and do data clearning

ntrain = df_train.shape[0]
ntest = df_test.shape[0]

y_train = df_train.SalePrice.values
y_train_final = copy.deepcopy(y_train)

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True) #drop SalePrice

all_data.shape #Checking dataframe size

#Identify features with missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})


#Impute based on description.txt
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["GarageQual"] = all_data["GarageQual"].fillna("None")
all_data["GarageCond"] = all_data["GarageCond"].fillna("None")
all_data["GarageFinish"] = all_data["GarageFinish"].fillna("None")
all_data["GarageType"] = all_data["GarageType"].fillna("None")
all_data["BsmtExposure"] = all_data["BsmtExposure"].fillna("None")
all_data["BsmtCond"] = all_data["BsmtCond"].fillna("None")
all_data["BsmtQual"] = all_data["BsmtQual"].fillna("None")
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].fillna("None")
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].fillna("None")
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["Functional"] = all_data["Functional"].fillna("Typ")


#ASSUMPTION ************
#LotFrontage -> fill in missing values by the median LotFrontage of the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#************


#ASSUMPTION ************
#GarageYrBlt, GarageArea, GarageCars
#For these categorical garage-related NA means that there is no garage
all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(0)
all_data["GarageArea"] = all_data["GarageArea"].fillna(0)
all_data["GarageCars"] = all_data["GarageCars"].fillna(0)


#BsmtFullBath, BsmtHalfBath, BsmtUnfSF, TotalBsmtSF, BsmtFinSF1, BsmtFinSF2
#For these categorical basement-related NA means that there is no basement
all_data["BsmtFullBath"] = all_data["BsmtFullBath"].fillna(0)
all_data["BsmtHalfBath"] = all_data["BsmtHalfBath"].fillna(0)
all_data["BsmtUnfSF"] = all_data["BsmtUnfSF"].fillna(0)
all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(0)
all_data["BsmtFinSF2"] = all_data["BsmtFinSF2"].fillna(0)
all_data["BsmtFinSF1"] = all_data["BsmtFinSF1"].fillna(0)

#MasVnrArea NA means no masonry veneer for these houses, therefore We can fill 0
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
#************

#MSSubClass -> NA means No building class. We can replace missing values with None
all_data["MSSubClass"] = all_data["MSSubClass"].fillna("None")

#************

#ASSUMPTION *************

#MSZoning -> There is only 4 NA value, lets compute this with most comon value
# of the entire dataframe.
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#Utilities -> There is only 2 NA value, lets compute this with most comon value
# of the entire dataframe.
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])

#Electrical -> There is only 1 NA value, lets compute this with most comon value
# of the entire dataframe.
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

#KitchenQual -> There is only 1 NA value, lets compute this with most comon value
# of the entire dataframe.
#KitchenQual -> data description says NA means "None".
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

#Exterior1st -> There is only 1 NA value, lets compute this with most comon value
# of the entire dataframe.
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

#Exterior2nd -> There is only 1 NA value, lets compute this with most comon value
# of the entire dataframe.
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

#SaleType -> There is only 1 NA value, lets compute this with most comon value
# of the entire dataframe.
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#************

print("X train data")
print(xtrain)

print("Y train data")
print(y_train)

print("test data")
print(xtest)

