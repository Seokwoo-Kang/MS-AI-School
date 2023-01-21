# HuggingFace Transformers를 활용한 문장 분류 모델 학습
"""
klue/roberta-base 모델을 KLUE 내 NLI 데이터셋을 활용하여 모델을 훈련하는 예제를 다루게 됩니다.

학습을 통해 얻어질 klue-roberta-base-nli 모델은 입력된 두 문장의 추론 관계를 예측하는데 사용할 수 있게 됩니다.

학습 과정 이후에는 간단한 예제 코드를 통해 모델이 어떻게 활용되는지도 함께 알아보도록 할 것입니다.

모든 소스 코드는 huggingface-notebooks를 참고하였습니다.
"""

import random
import logging

import numpy as np
import pandas as pd
# import datasets >> pip install -U transformers datasets scipy scikit-learn
import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_checkpoint = "klue/roberta-base"
batch_size = 128
task = "nli" # Task: Natural Language Inference (NLI), Metrics: Accuracy

# 이제 HuggingFace datasets 라이브러리에 등록된 KLUE 데이터셋 중, NLI 데이터를 내려받습니다.
datasets = load_dataset("klue", task)
print(datasets["train"][0])
# KLUE NLI는 entailment = ~ 따라 ?? ,
# neutral 애매한 그리고
# contradiction 모순 세 개의 라벨을 지니는 데이터셋임을 확인할 수 있습니다.

"""
훈련 과정 중 모델의 성능을 파악하기 위한 메트릭을 설정합니다.
datasets 라이브러리에는 이미 구현된 메트릭을 사용할 수 있는 load_metric 함수가 있습니다.
그 중 GLUE 데이터셋에 이미 다양한 메트릭이 구현되어 있으므로, 
GLUE 그 중에서도 KLUE NLI와 동일한 accuracy 메트릭을 사용하는 qnli 태스크의 메트릭을 사용합니다.
"""
# accuracy 메트릭이 정상적으로 작동하는지 확인하기 위해, 랜덤한 예측 값과 라벨 값을 생성합니다.
metric = load_metric("glue", "qnli")
fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
print(fake_preds, fake_labels)

test = metric.compute(predictions=fake_preds, references=fake_labels)
print(test)

# 이제 학습에 활용할 토크나이저를 로드해오도록 합니다.
"""
토크나이저는 NLP 파이프라인의 핵심 구성 요소 중 하나입니다. 토크나이저는 단지 1가지 목적을 가지고 있습니다. 
즉, 입력된 텍스트를 모델에서 처리할 수 있는 데이터로 변환하는 것입니다. 모델은 숫자만 처리할 수 있으므로, 
토크나이저는 텍스트 입력을 숫자 데이터로 변환해야 합니다. 
"""
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# 로드된 토크나이저가 두 개 문장을 토큰화하는 방식을 파악하기 위해 두 문장을 입력 값으로 넣어줘보도록 합시다.
token_test = tokenizer("힛걸 진심 최고로 멋지다.", "힛걸 진심 최고다 그 어떤 히어로보다 멋지다")
print(token_test)

"""
{'input_ids': [0, 3, 7254, 3841, 2200, 11980, 2062, 18, 2, 3, 7254, 3841, 2062, 636, 3711, 12717, 2178, 2062, 11980, 2062, 2], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""

"""
이제 앞서 로드한 데이터셋에서 각 문장에 해당하는 value 를 뽑아주기 위한 key 를 정의합니다.
앞서 KLUE NLI 데이터셋의 두 문장은 각각 premise와 hypothesis라는 이름으로 정의된 것을 확인하였으니, 
두 문장의 key 는 마찬가지로 각각 premise, hypothesis가 되게 됩니다.
"""
sentence1_key, sentence2_key = {"premise", "hypothesis"}
# print(f"Sentence 1 : {datasets['train'][0][sentence1_key]}")

"""
이제 key 도 확인이 되었으니, 데이터셋에서 각 예제들을 뽑아와 토큰화 할 수 있는 함수를 아래와 같이 정의해줍니다.
해당 함수는 모델을 훈련하기 앞서 데이터셋을 미리 토큰화 시켜놓는 작업을 위한 콜백 함수로 사용되게 됩니다.
인자로 넣어주는 truncation는 모델이 입력 받을 수 있는 최대 길이 이상의 토큰 시퀀스가 들어오게 될 경우, 
최대 길이 기준으로 시퀀스를 자르라는 의미를 지닙니다.
( * return_token_type_ids는 토크나이저가 token_type_ids를 반환하도록 할 것인지를 결정하는 인자입니다.
 transformers==4.7.0 기준으로 token_type_ids가 기본적으로 반환되므로 token_type_ids 자체를 사용하지 않는
  RoBERTa 모델을 활용하기 위해 해당 인자를 False로 설정해주도록 합니다.)
"""

def preprocess_function(examples):
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        return_token_type_ids = False,

    )

# 이제 정의된 전처리 함수를 활용해 데이터셋을 미리 토큰화시키는 작업을 수행합니다.
# datasets 라이브러리를 통해 얻어진 DatasetDict 객체는 map() 함수를 지원하므로,
# 정의된 전처리 함수를 데이터셋 토큰화를 위한 콜백 함수로 map() 함수 인자로 넘겨주면 됩니다.
# 보다 자세한 내용은 문서를 참조해주시면 됩니다.
# >> https://huggingface.co/docs/datasets/process#processing-data-with-map
encoded_datasets = datasets.map(preprocess_function, batched=True)

"""
학습을 위한 모델을 로드할 차례입니다.
앞서 살펴본 바와 같이 KLUE NLI에는 총 3개의 클래스가 존재하므로, 
3개의 클래스를 예측할 수 있는 SequenceClassification 구조로 모델을 로드하도록 합니다.
"""
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

"""
모델을 로드할 때 발생하는 경고 문구는 두 가지 의미를 지닙니다.
Masked Language Modeling 을 위해 존재했던 lm_head가 현재는 사용되지 않고 있음을 의미합니다.
문장 분류를 위한 classifier 레이어를 백본 모델 뒤에 이어 붙였으나 아직 훈련이 되지 않았으므로, 
학습을 수행해야 함을 의미합니다.
"""

"""
마지막으로 앞서 정의한 메트릭을 모델 예측 결과에 적용하기 위한 함수를 정의합니다.
입력으로 들어오는 eval_pred는 EvalPrediction 객체이며, 모델의 클래스 별 예측 값과 정답 값을 지닙니다.
클래스 별 예측 중 가장 높은 라벨을 argmax()를 통해 뽑아낸 후, 정답 라벨과 비교를 하게 됩니다.
"""
def compute_metrics(eval_pred) :
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 각 인자에 대한 자세한 설명은 문서에서 참조해주시면 됩니다.
# >>> https://huggingface.co/docs/transformers/main_classes/trainer#trainingarguments
metric_name = "accuracy"

args = TrainingArguments(
    "test-nli",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy='epoch',
    load_best_model_at_end= True,
    metric_for_best_model=metric_name,
)

"""
이제 로드한 모델, 인자 관리 클래스, 데이터셋 등을 Trainer 클래스를 초기화에 넘겨주도록 합니다.

(TIP: Q: 이미 encoded_datasets을 만드는 과정에 토큰화가 이루어졌는데 토크나이저를 굳이 넘겨주는 이유가 무엇인가요?,
A: 토큰화는 이루어졌지만 학습 과정 시, 데이터를 배치 단위로 넘겨주는 과정에서 배치에 포함된 가장 긴 시퀀스 기준으로 truncation을 수행하고 최대 길이 시퀀스 보다 짧은 시퀀스들은 그 길이만큼 padding을 수행해주기 위함입니다.)
"""
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
metrics = trainer.evaluate(encoded_datasets["validation"])
print(metrics)