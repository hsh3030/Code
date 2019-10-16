#  모듈을 불러온다
import pymssql as ms
import numpy as np

# 데이터 베이스에 연결
conn = ms.connect(server='localhost', user = 'bit', password='1234', database='bitdb')

# 커서를 만든다
cursor = conn.cursor()

# 커서에 쿼리를 입력해 실행 시킨다.
cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()
print(row)
conn.close()

# numpy 변환 및 저장
a = np.asarray(row)
print(a)
print(a.shape)
print(type(a))
np.save('test_a.npy', a)

'''
# 한행을 가져옵니다.
row = cursor.fetchone()
print(type(row))

while row:
    # print("첫컬럼=%s, 둘컬럼=%s" %(row[0], row[1]))
    print(row)
    row = cursor.fetchone()

# 연결을 닫습니다.
conn.close()
'''