# from imputation1 import *
import os, sys

#'/home/user/example/parent/child'
current_path = os.path.abspath('.')

from reduct_and_immute import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from math import sqrt
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
import copy

import matplotlib.pyplot as plt  # Matlab-style plotting
# %matplotlib inline

# DATA DUMMIFICATION
all_data_dumify = pd.get_dummies(data=all_data, drop_first=True)
all_data_dumify.shape
# SPLITTING DATA
xtrain = all_data_dumify[:ntrain]
xtest = all_data_dumify[ntrain:]
all_data.shape

xtrain_no_dummify = all_data[:ntrain]
xtest_no_dummify = all_data[ntrain:]
xtest_no_dummify.head()
# xtrain_no_dummify["BldgType"] = lb_make.fit_transform(xtrain_no_dummify["BldgType"])


# Encode All Categorical Data For Decision TreeRegression
char_cols = xtrain_no_dummify.dtypes.pipe(lambda x: x[x == 'object']).index
xtest_no_dummify.dtypes

for c in char_cols:
    xtrain_no_dummify[c] = pd.factorize(xtrain_no_dummify[c])[0]

char_cols = xtest_no_dummify.dtypes.pipe(lambda x: x[x == 'object']).index

# Encode All Categorical Data For Decision TreeRegression
for c in char_cols:
    xtest_no_dummify[c] = pd.factorize(xtest_no_dummify[c])[0]
# xtrain_no_dummify.dtypes
# xtrain_no_dummify["Electrical"]
xtest_no_dummify.dtypes
xtest_no_dummify.head()

df_train = all_data_dumify[:ntrain]
df_test = all_data_dumify[ntest:]

X = df_train.loc[:, ~df_train.columns.isin(['SalePrice'])] # Remove Specific column by name

y = copy.deepcopy(y_train)
y_train_final = copy.deepcopy(y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# *****************************************************************************************************************************************************************************
# *****************************************************************************************************************************************************************************
# #random forest

# n_estimators = range(1,100)
# #[int(x) for x in np.linspace(start = 1, stop = 100, num = 1)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# #cretion
# criterion=['mse']
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
               
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap
#                }
# pprint(random_grid)


# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(xtrain, y_train_final)

# #bset params
# rf_random.best_params_

X_no_dummify = xtrain_no_dummify.loc[:, ~xtrain_no_dummify.columns.isin(['SalePrice'])]
X_train_no_dummify, X_test_no_dummify, y_train_no_dummify, y_test_no_dummify = train_test_split(X_no_dummify, y, test_size=0.2, random_state=42)

# xtrain_no_dummify
rf = RandomForestRegressor()
rf = RandomForestRegressor(n_estimators = 94,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',max_depth=70,bootstrap=False)

#fit with best params of the train data
rf.fit(X_train_no_dummify, y_train_no_dummify)

#score with train data
rf.score(X_train_no_dummify, y_train_no_dummify)
#predict
y_predicted=rf.predict(X_test_no_dummify)
# fit test data with npredicted
rf.fit(X_test_no_dummify,y_predicted)
#check score
rf.score(X_test_no_dummify,y_predicted)

#select the most impotant features in random forest
pd.set_option('display.max_rows', None)
X_train_no_dummify.shape
important_features = pd.Series(data=rf.feature_importances_,index=X_train_no_dummify.columns)
important_features.sort_values(ascending=False,inplace=True)
important_features

#plot graph of most import feature
important_features.plot(kind = 'bar')

#lasso model
alphas = np.arange(0,10)
grid = GridSearchCV( estimator=Lasso(), param_grid = {'alpha':alphas} )
grid.fit(X_train, y_train)
lasso_clf = grid.best_estimator_
#best lambda
lasso_clf
#set best lambda and fit train data
lasso = Lasso()
lasso.set_params(alpha = 9.0)
lasso.fit(X_train, y_train)
lasso.score(X_train, y_train)
#get cofficient
lasso.coef_
#predicted value from train data
predicted_y1=lasso.predict(xtest)
#score of the predicted data
lasso.score(xtest, predicted_y1)

#ridge model
alphas = np.arange(0,10)
grid = GridSearchCV( estimator=Ridge(), param_grid = {'alpha':alphas} )
grid.fit(X_train, y_train)
ridge_clf = grid.best_estimator_
#best lambda
ridge_clf
#set best lambda and fit train data
ridge = Ridge()
ridge.set_params(alpha = 7.0)
ridge.fit(X_train, y_train)
ridge.score(X_train, y_train)
#get cofficient
ridge.coef_
#predicted value from train data
predicted_y1=ridge.predict(X_test)
#score of the predicted data
ridge.score(X_test, predicted_y1)



# ******************************************************************************************************************************************************************************
# ******************************************************************************************************************************************************************************
# model#1 Lasso
lasso = Lasso(alpha=0.01, max_iter=1000)
lasso_0_05 = Lasso(alpha=0.05, max_iter=1000)

#lasso = Lasso(alpha=0.01, max_iter=10e5)
# lasso.fit(train, y_train)

# Linear Regression
linearreg = LinearRegression()
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Gradient Boosting
gbm = GradientBoostingRegressor()
gbm = gbm.fit(np.nan_to_num(X_train),y_train)
# print(gbm.score(np.nan_to_num(X),y))


# Random Forest
rndfrst = RandomForestRegressor(n_estimators = 62,max_features='sqrt', random_state = 0, verbose=0) 

rndfrst.fit(X_train, y_train)

modelList = [linearreg, gbm, lasso,lasso_0_05, rndfrst]
modelSeries = pd.Series(modelList, index=['Linear Regression', 'Gradient Boosting','lasso','lasso_0_05','rndfrst'])

# fit all the models to the training data
modelSeries.apply(lambda t:t.fit(X_train, y_train))


# calculate the train/test accuracy
ans = modelSeries.apply(lambda t:pd.Series([t.score(X_train, y_train), t.score(X_test, y_test)]))
ans.columns = ['train score', 'test score']
print(ans)


# Prediction after model selection
# y_pred.regressor = regressor.predict(X_test)
(df_train.isnull().sum())
df_test1 = df_test.loc[:, ~df_test.columns.isin(['SalePrice','id'])] # Remove Specific column by name

y_pred_train = gbm.predict(df_train)
y_pred_train

y_pred = gbm.predict(df_test)
y_pred


test_ID


  # create regressor object 
# rndfrst = RandomForestRegressor(n_estimators = 100, random_state = 0) 


rndfrst.score(X_train, y_train)

y_pred_final = rndfrst.predict(df_train)

y_pred_rndfrst = rndfrst.predict(df_test)
y_pred_rndfrst

# GridSearch
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import chi2_contingency

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC

# from subprocess import check_output

# from sklearn.model_selection import GridSearchCV


print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred_final, y_pred_rndfrst))  
print('Mean Squared Error:', metrics.mean_squared_error(y_pred_final, y_pred_rndfrst))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_pred_final, y_pred_rndfrst)))

y_pred.shape

y_train.shape

y_pred_final = rf.predict(xtest_no_dummify)
y_pred_final.shape

# Prepare Submission File
# ensemble = stacked_pred *1
submit = pd.DataFrame()
submit['id'] = test_ID
# submit['SalePrice'] = ensemble
submit['SalePrice'] = pd.DataFrame(y_pred_final)
# ----------------------------- Create File to Submit --------------------------------
# submit.to_csv('SalePrice_N_submission.csv', index = False)
submit.to_csv('./submission/SalePrice_N_submission10.csv', index = False)

submit.head()

important_features.plot(kind = 'bar')
plt.show()

