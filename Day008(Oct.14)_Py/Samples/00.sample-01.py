import random

random_number = random.randint(1, 100)

#print(random_number)

game_count = 1

while True:
    #문자입력시 에러 잡기
    try:

        my_number = int(input("1-100 사이의 숫자를 입력하세요 : ")) 
        #input으로 받은 문자열을 int를 통해 정수형으로 my_number에 저장

        if my_number > random_number:
            print('Please Down!!')
        elif my_number < random_number:
            print('Please Up!!')
        else:
            print(f'축하합니다. {game_count} 번만에 맞추셨습니다.')
            #f 를 맨앞에 적어 .format 과 같은 기능( 내부 값 호출???)
            break

        game_count = game_count + 1


