import socket
import time
HOST ='192.168.0.158'
PORT = 9009
PORT2 = 9010
 
def getFileFromServer(filename):
    data_transferred = 0
 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST,PORT))
        sock.sendall(filename.encode())
 
        data = sock.recv(1024)
        if not data:
            print('파일[%s]: 서버에 존재하지 않거나 전송중 오류발생' %filename)
            return
 
        with open('./download/' + filename,'wb') as f:
            try:
                while  data:
                    f.write(data)
                    data_transferred += len(data)
                    data = sock.recv(1024)
            except Exception as e:
                print(e)
def postFiletoServer(filename):
    data_transferred = 0
 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST,PORT2))
        sock.sendall(filename.encode())
        
        # data = sock.recv(1024)
        # print(1)
        # data = data.decode()

        # received = int(data)
        # if received == 1010:
        #     print("같은 이름의 데이터가 존재합니다.")
        #     return
        # print(1)
        time.sleep(1)
        with open('./upload/' + filename,'rb') as f:           
            try:
                data = f.read(1024)
                while len(data):
                    sock.send(data)
                    data_transferred += len(data)
                    data = f.read(1024)



            # l = f.read(1024)
            # while (l):
            #     print('Sending...')
            #     s.send(l)
            #     l = f.read(1024)
            # f.close()
            # print("Done Sending")
            except Exception as e:
                print(e)
    print('파일[%s] 전송종료. 전송량 [%d]' %(filename, data_transferred))

clientMode = int(input("모드를 선택하세요: 1. 파일다운로드, 2.파일업로드"))
if clientMode == 1:
    filename = input('다운로드 받은 파일이름을 입력하세요:')
    getFileFromServer(filename)
elif clientMode == 2:
    filename = input('업로드할 파일이름을 입력하세요:')
    postFiletoServer(filename)
        
