# pip install transformers
import pandas as pd
from transformers import pipeline
# pipeline() 함수를 호출하면서 관심 작업 이름을 전달해 파이프라인 객체를 생성
# https://huggingface.co/docs/transformers/main/en/pipeline_tutorial#pipeline-usage
# 다양한 파이프라인 정보 >> https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#pipelines
classifier = pipeline("text-classification")

# 처음 이 코드를 실행하면 파이프라인이 자동으로 허깅페이스 허브에서 모델 가중치 다운로드 합니다.
# 파이프라인 객체를 다시 만들 때는 가중치가 이미 다운로드됐으므로 캐싱된 버전을 사용한다는 안내 메시지가 나옵니다.
# 기본적으로 txt-classification 파이프라인은 감성 분석을 위해 설계된 모델을 사용하지만, 다중 분류와 다중 레이블 분류도 지원합니다.

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I"""

outputs = classifier(text)
print(outputs)
# 모델은 텍스트가 부정적이라고 확신합니다. 감성 분석 작업에서 파이프라인은 POSITIVE 와 NEGATIVE 레이블중 하나를 반환

# NER
ner_tagger = pipeline("ner", aggregation_strategy='simple')
outputs = ner_tagger(text)
temp = pd.DataFrame(outputs)
print(temp)

reder = pipeline("question-answering")
question = "What does the customer want ?"
output = reder(question=question, context=text)
temp1 = pd.DataFrame([output])
print(temp1)

summarizer = pipeline("summarization")
output = summarizer(text, max_length=60, clean_up_tokenization_spaces=True)
print(output[0]['summary_text'])

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
print(pipe('こんにちは'))

from transformers import set_seed
set_seed(888)

generator = pipeline("text-generation")
response = "Dear Bumlebee , I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response : \n" + response
output = generator(prompt, max_length=200)
print(output[0]['generated_text'])