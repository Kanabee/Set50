import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score ,
                                       mean_squared_error,
                                      mean_absolute_error,
                                       mean_absolute_percentage_error )
import warnings
warnings.filterwarnings('ignore')
np.random.seed(12345)

from google.colab import files
upload = files.upload()

data.drop(['Date','Low_0'],axis=1 , inplace=True)
data

data.info()



target_name ='High_0'
feature_name = list(data.columns.drop(target_name))
X = data[feature_name]
y = data[target_name]
X_train , X_test ,y_train ,y_test = train_test_split(X,y,test_size = 0.2,shuffle =False)

reg = LinearRegression()
reg.fit(X_train,y_train)

reg.intercept_

reg.coef_

X_train

y_pred_train = reg.predict(X_train)

y_pred_test = reg.predict(X_test)

print('r2_score =\t\t\t' , r2_score(y_train,y_pred_train))
print('mean_squared_error = \t\t', mean_squared_error(y_train,y_pred_train))
print('mean_absolute_erroe = \t\t', mean_absolute_error(y_train,y_pred_train))
print('mean_absolute_percentage_error =' , mean_absolute_percentage_error(y_train,y_pred_train))

print('r2_score =\t\t\t' , r2_score(y_test,y_pred_test))
print('mean_squared_error = \t\t', mean_squared_error(y_test,y_pred_test))
print('mean_absolute_erroe = \t\t', mean_absolute_error(y_test,y_pred_test))
print('mean_absolute_percentage_error =' , mean_absolute_percentage_error(y_test,y_pred_test))
