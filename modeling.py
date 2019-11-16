from imputation1 import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from math import sqrt

# DATA DUMMIFICATION
all_data_dumify = pd.get_dummies(data=all_data, drop_first=True)

# SPLITTING DATA
xtrain = all_data_dumify[:ntrain]
xtest = all_data_dumify[ntrain:]

xtrain_no_dummify = all_data[:ntrain]
xtest_no_dummify = all_data[ntrain:]


all_data
df_train = all_data_dumify[:ntrain]
df_test = all_data_dumify[ntest:]

X = df_train.loc[:, ~df_train.columns.isin(['SalePrice'])] # Remove Specific column by name

y = y_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = xtrain
# X_test = xtest
# y_test
# y_train = 1

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

modelList = [linearreg, gbm, lasso,lasso_0_05]
modelSeries = pd.Series(modelList, index=['Linear Regression', 'Gradient Boosting','lasso','lasso_0_05'])

# fit all the models to the training data
modelSeries.apply(lambda t:t.fit(X_train, y_train))


# calculate the train/test accuracy
ans = modelSeries.apply(lambda t:pd.Series([t.score(X_train, y_train), t.score(X_test, y_test)]))
ans.columns = ['train score', 'test score']
print(ans)


# Prediction after model selection
# y_pred.regressor = regressor.predict(X_test)
print(df_train.isnull().sum())
df_test1 = df_test.loc[:, ~df_test.columns.isin(['SalePrice','id'])] # Remove Specific column by name

y_pred_train = gbm.predict(df_train)
y_pred_train

y_pred = gbm.predict(df_test)
y_pred


test_ID

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor 
  # create regressor object 
# rndfrst = RandomForestRegressor(n_estimators = 100, random_state = 0) 

rndfrst = RandomForestRegressor(n_estimators = 62,max_features='sqrt', random_state = 0, verbose=2) 

# 'criterion': 'mae',
#  'max_depth': 8,
#  'max_features': 'auto',
#  'n_estimators': 100
 
rndfrst.fit(X_train, y_train)
rndfrst.score(X_train, y_train)

y_pred_final = rndfrst.predict(df_train)

y_pred_rndfrst = rndfrst.predict(df_test)
y_pred_rndfrst

# GridSearch
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.model_selection import GridSearchCV

# rfr = RandomForestRegressor(verbose=2)

# param_grid = {
#     'n_estimators': [62],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     # 'max_depth' : [1,5,10,15,20,25,'auto'],
#     'criterion' :['mse', 'mae']
# }

# # param_grid = {
# #   # 'bootstrap': True,
# #  'criterion': 'mse',
# # #  'max_depth': [None],
# #  'max_features': 'auto',
# # #  'max_leaf_nodes': None,
# #  'min_impurity_decrease': 0.0,
# # #  'min_impurity_split': None,
# #  'min_samples_leaf': 1,
# #  'min_samples_split': 2,
# #  'min_weight_fraction_leaf': 0.0,
# #  'n_estimators': range(1, 100),
# #  'n_jobs': 12,
# #  'oob_score': False,
# #  'random_state': 42,
# #  'verbose': 0,
# #  'warm_start': False
# #  }

# CV_rfr = GridSearchCV(estimator=rfr, param_grid=param_grid, cv= 5, n_jobs = 10)
# CV_rfr.fit(X_train, y_train)

# CV_rfr.best_params_


# for j in range(1000):

#             X_train, X_test, y_train, y_test = train_test_split(X, y , random_state =j,     test_size=0.35)
#             lr = LarsCV().fit(X_train, y_train)

#             tr_score.append(lr.score(X_train, y_train))
#             ts_score.append(lr.score(X_test, y_test))

#         J = ts_score.index(np.max(ts_score))

#         X_train, X_test, y_train, y_test = train_test_split(X, y , random_state =J, test_size=0.35)
#         M = LarsCV().fit(X_train, y_train)
#         y_pred = M.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred_final, y_pred_rndfrst))  
print('Mean Squared Error:', metrics.mean_squared_error(y_pred_final, y_pred_rndfrst))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_pred_final, y_pred_rndfrst)))

y_pred.shape

y_train.shape



# Prepare Submission File
# ensemble = stacked_pred *1
submit = pd.DataFrame()
submit['id'] = test_ID
# submit['SalePrice'] = ensemble
submit['SalePrice'] = pd.DataFrame(y_pred_rndfrst)
# ----------------------------- Create File to Submit --------------------------------
# submit.to_csv('SalePrice_N_submission.csv', index = False)
# submit.to_csv('SalePrice_N_submission4.csv', index = False)

submit.head()