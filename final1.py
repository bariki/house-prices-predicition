import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.linear_model import Lasso

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# color = sns.color_palette()
# sns.set_style('darkgrid')

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

#Explorer Outliers
fig, ax = plt.subplots()
ax.scatter(x = df_train['LotArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('LotArea', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = df_train['LotFrontage'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('LotFrontage', fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


#It seems there are many outliers in different variables therefore
#we need to make a robust model that can manage them, therefore lets study
#our target variable SalePrice


#Analysis on SalePrice

df_train['SalePrice'].describe()

sns.distplot(df_train['SalePrice'] , fit=norm)
#seems there is some deviate from the normal distribution, therefore
#we need to transform this variable and make it more normally distributed


sns.distplot( np.log1p(df_train["SalePrice"]) , fit=norm)
#now data appeared to be normal distributed hence lets apply this function
#to the data set


#Feature Analysis

df_train.info()
df_train.describe()
df_train.describe(include =np.object)


#checking for missing values in our dataframe
df_train.isnull().values.any()
df_train.isnull().sum()
df_train.isnull().sum().sum()

df_test.isnull().values.any()
df_test.isnull().sum()
df_test.isnull().sum().sum()


#We find out there is missing data, therefore lets combine there dataframe
#to big one and do data clearning

ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True) #drop SalePrice

all_data.shape #Checking dataframe size


#Identify features with missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data
all_data_na*len(all_data)/100

#Visualize features with missing data
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


#Visualizing data correlation
corrmat = df_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


#Imputing missing values

#PoolQC -> data description says NA means "No pool".
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

#MiscFeature -> data description says NA means "None".
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

#Alley -> data description says NA means "No alley access".
all_data["Alley"] = all_data["Alley"].fillna("None")

#Fence -> data description says NA means "None".
all_data["Fence"] = all_data["Fence"].fillna("None")

#FireplaceQu -> data description says NA means "None".
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#LotFrontage -> fill in missing values by the median LotFrontage of the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#GarageQual -> data description says NA means "None".
all_data["GarageQual"] = all_data["GarageQual"].fillna("None")

#GarageCond -> data description says NA means "None".
all_data["GarageCond"] = all_data["GarageCond"].fillna("None")

#GarageFinish -> data description says NA means "None".
all_data["GarageFinish"] = all_data["GarageFinish"].fillna("None")

#GarageType -> data description says NA means "None".
all_data["GarageType"] = all_data["GarageType"].fillna("None")


#ASSUMPTION ************
#GarageYrBlt, GarageArea, GarageCars
#For these categorical garage-related NA means that there is no garage
all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(0)
all_data["GarageArea"] = all_data["GarageArea"].fillna(0)
all_data["GarageCars"] = all_data["GarageCars"].fillna(0)
#************


#BsmtExposure -> data description says NA means "None".
all_data["BsmtExposure"] = all_data["BsmtExposure"].fillna("None")

#BsmtCond -> data description says NA means "None".
all_data["BsmtCond"] = all_data["BsmtCond"].fillna("None")

#BsmtQual -> data description says NA means "None".
all_data["BsmtQual"] = all_data["BsmtQual"].fillna("None")

#BsmtFinType1 -> data description says NA means "None".
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].fillna("None")

#BsmtFinType2 -> data description says NA means "None".
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].fillna("None")


#ASSUMPTION ************
#BsmtFullBath, BsmtHalfBath, BsmtUnfSF, TotalBsmtSF, BsmtFinSF1, BsmtFinSF2
#For these categorical basement-related NA means that there is no basement
all_data["BsmtFullBath"] = all_data["BsmtFullBath"].fillna(0)
all_data["BsmtHalfBath"] = all_data["BsmtHalfBath"].fillna(0)
all_data["BsmtUnfSF"] = all_data["BsmtUnfSF"].fillna(0)
all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(0)
all_data["BsmtFinSF2"] = all_data["BsmtFinSF2"].fillna(0)
all_data["BsmtFinSF1"] = all_data["BsmtFinSF1"].fillna(0)
#************

#MasVnrType -> data description says NA means "None".
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

#ASSUMPTION *************
#MasVnrArea NA means no masonry veneer for these houses, therefore We can fill 0
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
#************

#Functional -> data description says NA means "typical".
all_data["Functional"] = all_data["Functional"].fillna("Typ")

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

#ASSUMPTION *************
#MSSubClass -> NA means No building class. We can replace missing values with None
all_data["MSSubClass"] = all_data["MSSubClass"].fillna("None")
#************


all_data
all_data.shape

all_data_dumify = pd.get_dummies(data=all_data, drop_first=True)

all_data_dumify


train = all_data_dumify[:ntrain]
test = all_data_dumify[ntrain:]

df1 = train
df1['y_train'] = y_train
df1['y_train']

train.to_csv('dumified_train.csv')
test.to_csv('dumified_test.csv')
df1.to_csv('df1.csv')

#simple
model = LinearRegression()

train
test

y_train

model.fit(train, y_train)
model.score(train, y_train)
model.predict(test)

#lasso
lasso = Lasso(alpha=0.01, max_iter=1000)
#lasso = Lasso(alpha=0.01, max_iter=10e5)
lasso.fit(train, y_train)
lasso.score(train, y_train)

lasso.predict(test)


model2 = GradientBoostingRegressor()
model2 = model2.fit(train,y_train)
model2.score(train,y_train)

