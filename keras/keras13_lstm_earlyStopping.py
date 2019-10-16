# RNN 에 LSTM 은 포함된 상태
from numpy import array # as np 대신 바로 array 가져와 쓴다
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 만들기
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],[20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

# reshape 작업
x = x.reshape((x.shape[0], x.shape[1],1)) # x.shape[0] = 4행 , x.shape[1] = 3열 , 1 = 자르는 갯수 // y.shape는 결과값의 갯수로 생각 (4,)
print("x.shape : ", x.shape)

# 2. Model 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) # (3,1) ?행 3열 dim값 = 1
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()
# 3. 훈련 실행 (lstm에서는 layer과 node의 수 보다 epoch를 더 할 수록 결과값이 좋을 수 있다.)
model.compile(optimizer='adam', loss='mse')
#loss =>
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
#early_stopping => 개선할 여지가 없으면 중단시킨다.
model.fit(x, y, epochs = 10000, verbose=1, callbacks=[early_stopping])

x_input = array([25,35,45]) # 1,3, ????
x_input = x_input.reshape((1,3,1)) 

yhat = model.predict(x_input, verbose=1)
print(yhat)


# homework 왜 480 일까?
