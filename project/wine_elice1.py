# from elice_utils import EliceUtils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

train = pd.read_csv('./data/Ewine_train.csv')
test = pd.read_csv('./data/Ewine_test.csv')
# dataframe = DataFrame['Formatted Date']
train = train.fillna(0)
test = test.fillna(0)

# train = train[:10000]
sample = pd.DataFrame()
# sample["index"] = test['index']
x_train = train[['price']]

train_label = np.array(train["points"])

test = test[['price']]

x_train = np.array(x_train)

test = np.array(test)

x = x_train
y = test
# x = train.reshape(train.shape[0],train.shape[1])
# y = test.reshape(test.shape[0],test.shape[1])

x_train, x_test, y_train, y_test = train_test_split(x, train_label, random_state=1, test_size=0.2)

# tree = RandomForestClassifier(n_estimators = 2000,  n_jobs=-1, max_features = "auto", max_depth= 30, random_state=0)
# tree.fit(x_train, y_train)
parameters = {
    "max_depth": [3, 6, 10, 20, 30, 50, 100, 200, 300],  "gamma":[0.001, 0.0001], "booster": ['gbtree'],
    "loss": ['deviance', 'exponential'], "estimator": [100, 200, 300, 400, 500],
    "importance_type": ['gain'], "n_jobs": [10], "base_score": [0.3, 0.5], "reg_alpha": [0, 1, 2, 3, 4]
}
# 직선 회귀 분석하기
kfold_cv = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold_cv)
model.fit(x_train, y_train)
print("최적의 매개 변수 = ", model.best_estimator_)

y_predict = model.predict(y) # predict : 예측치 확인

print(len(sample))
print(y_predict.shape)
sample["points"] = y_predict

sample.to_csv("visi_sample.csv", index=False)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, y_predict))
