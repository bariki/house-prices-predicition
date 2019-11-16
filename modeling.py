# from imputation1 import *
from reduct_and_immute import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from math import sqrt
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

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

# *****************************************************************************************************************************************************************************
# *****************************************************************************************************************************************************************************
#random forest

n_estimators = range(1,100)
#[int(x) for x in np.linspace(start = 1, stop = 100, num = 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#cretion
criterion=['mse']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
               }
pprint(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(xtrain, y_train_final)

#bset params
rf_random.best_params_


rf = RandomForestRegressor(n_estimators = 94,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',
                           max_depth=70,bootstrap=False)

#fit with best params of the train data
rf.fit(X_train, y_train)


#score with train data
rf.score(X_train, y_train)

#predict
y_predicted=rf.predict(X_test)

# fit test data with npredicted
rf.fit(X_test,y_predicted)

#check score
rf.score(X_test,y_predicted)
# #checking error
# train_error = (1 - train_score)
# test_error = (1 - train_score)
# print("The training error is: %.5f" %train_error)
# print("The test     error is: %.5f" %test_error)






#lasso model

from sklearn.model_selection import GridSearchCV
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

from sklearn.model_selection import GridSearchCV
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

predicted_y1=ridge.predict(xtest)

#score of the predicted data

ridge.score(xtest, predicted_y1)



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
rndfrst = RandomForestRegressor(n_estimators = 62,max_features='sqrt', random_state = 0, verbose=2) 

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



print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred_final, y_pred_rndfrst))  
print('Mean Squared Error:', metrics.mean_squared_error(y_pred_final, y_pred_rndfrst))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_pred_final, y_pred_rndfrst)))

y_pred.shape

y_train.shape


y_pred_final = rf.predict(df_test)
y_pred_final.shape

# Prepare Submission File
# ensemble = stacked_pred *1
submit = pd.DataFrame()
submit['id'] = test_ID
# submit['SalePrice'] = ensemble
submit['SalePrice'] = pd.DataFrame(y_pred_final)
# ----------------------------- Create File to Submit --------------------------------
# submit.to_csv('SalePrice_N_submission.csv', index = False)
submit.to_csv('./submission/SalePrice_N_submission7.csv', index = False)

submit.head()