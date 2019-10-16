from elice_utils import EliceUtils
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

train = pd.read_csv('./data/elice_train.csv')
test = pd.read_csv('./data/elice_test.csv')
# dataframe = DataFrame['Formatted Date']
train = train.fillna(0)
test = test.fillna(0)

# train = train[:10000]
sample = pd.DataFrame()
# sample["index"] = test['index']
x_train = train[['price']]

train_label = np.array(train["points"])

test = test[['price']]

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
train = scaler.transform(x_train)
test = scaler.transform(test)

x_train = np.array(x_train)

test = np.array(test)

x = train.reshape(train.shape[0],train.shape[1],1)
y = test.reshape(test.shape[0],test.shape[1],1)
print(x.shape)
print(train_label.shape)
x_train, x_test, y_train, y_test = train_test_split(x, train_label, random_state=1, test_size=0.2)


# x_train = x_train.reshape(x_train.shape[1], x_train.shape[0])
# x_test = x_test.reshape(x_test.shape[1], x_test.shape[0])
# y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
# y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(LSTM(1024, input_shape=(6,1), activation = 'relu', return_sequences = True)) # input과 output값 변경
model.add(LSTM(1024, activation = 'relu'))
model.add(Dense(20480, activation = 'relu'))
# model.add(Dense(20048, activation = 'relu'))
# model.add(Dense(10024, activation = 'relu'))
# model.add(Dense(12, activation = 'relu'))
model.add(Dense(1, activation = 'relu')) 

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs = 20000, batch_size=1024)

y_predict = model.predict(y) # predict : 예측치 확인
# print(len(sample))
print(y_predict.shape)
sample["points"] = y_predict

sample.to_csv("visi_sample.csv", index=False)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test 와 y_predict 비교하기 위한 함수 (원래의 값과 예측값을 비교)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # 비교하여 그 차이를 빼준다
print("RMSE : ", RMSE(y_test, y_predict))
