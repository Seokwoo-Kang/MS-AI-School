import socket
        #서버이름을 통해 IP주소 할당   서버의 이름(컴퓨터이름)
in_addr = socket.gethostbyname(socket.gethostname())

print(in_addr)
