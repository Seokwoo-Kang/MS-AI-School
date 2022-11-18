# Docker(Ububtu)
    Azure(Virtual Machine-Ubuntu 22.04 LTS)

## Windows Terminal

```bash
~$ sudo apt-get update
~$ sudo apt-get upgrade
~$ sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
# 우분투 환경을 위한 패키지 설치
~$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# 우분투에서 도커를 사용하기 위한 키 연결과 설치?

~$ echo   "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

~$ sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io

~$ sudo docker run hello-world

~$ ps
~$ ps -al
~$ docker ps
~$ sudo docker ps
~$ sudo usermod -a -G docker $USER

~$ sudo service docker restart
~$ exit

~$ docker ps
~$ docker pull --help
~$ docker pull ubuntu:18.04
~$ docker images
~$ docker ps -a

~$ docker run --help
~$ docker run -it --name demo1 ubuntu:18.04 /bin/bash
~$ docker run -it --name demo2 ubuntu:18.04 /bin/bash
~$
~$
~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$

~$


 


```


