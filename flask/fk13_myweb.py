# 모듈 불러오기
import pyodbc as pyo

# 연결 문자열을 셋팅
server = 'localhost'
database = 'bitdb'
username = 'bit'
password = '1234'

# 데이터 베이스를 연결
cnxn = pyo.connect('DRIVER={ODBC Driver 13 for SQL Server}; SERVER=' +server+'; PROT=1433; DATABASE=' +database+';UID=' +username+';PWD=' +password)

# 커서를 만듭니다.
cursor = cnxn.cursor()

# 커서에 쿼리를 입력해 실행시킨다
tsql = "SELECT * FROM iris2;"

# flask 웹서버를 실행
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/sqltable")

def showsql():
    cursor.execute(tsql)
    return render_template('myweb.html', rows=cursor.fetchall())

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)

















with cursor.execute(tsql):
    # 한 행을 가져옵니다.
    row = cursor.fetchone()
    # 행이 존재할 때까지, 하나씩 행을 증가시키면서 모든 컬럼을 공백으로 구분해 출력
    while row:
        print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " + str(row[3]) + " " + str(row[4]))
        row = cursor.fetchone()
cnxn.close()