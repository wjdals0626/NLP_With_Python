#!/usr/bin/env python
# coding: utf-8

#패키지 import
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#데이터 불러오기 (첨부된 '신고 데이터')
data=pd.read_csv('C:/Users/pinkk/OneDrive/바탕 화면/jungmin data/22-2/통계학과자연어처리PBL/신고 데이터2.csv', encoding='CP949')
data

# 데이터 개수 확인
print('전체 데이터의 개수: {}'.format(len(data)))

#각 데이터에 대해 신고 길이 확인
train_length=data['신고'].apply(len)
train_length.head()


#전체 데이터의 길이에 대한 히스토그램 그리기
plt.figure(figsize=(12,5)) # 그래프에 대한 이미지 크기 선언
plt.title('Histogram of length of report') # 그래프 제목
plt.xlabel('Length of report') # 그래프 x 축 라벨
plt.ylabel('Number of report') # 그래프 y 축 라벨
plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word') # 히스토그램 선언

#boxplot을 그리기 위한 작업
print('신고 길이 최댓값: {}'.format(np.max(train_length)))
print('신고 길이 평균값: {:.2f}'.format(np.mean(train_length)))
print('신고 길이 표준편차: {:.2f}'.format(np.std(train_length)))
print('신고 길이 중간값: {}'.format(np.median(train_length)))

#boxplot을 그리기 위한 패키지 import
import matplotlib.pyplot as plt
import numpy as np

#학습 데이터 길이를 boxplot을 이용하여 시각화
plt.boxplot(train_length)
plt.show()

#패키지 install
get_ipython().system('pip install wordcloud')

#패키지 import
from konlpy.tag import Twitter
from collections import Counter

file = open('신고 데이터.csv')

lists = file.readlines()
new_lists = list(map(lambda s : s.strip(), lists))
file.close()

new_lists

twitter = Twitter()
morphs = [] 

for sentence in new_lists: 
    morphs.append(twitter.pos(sentence)) 
    
print(morphs)

noun_adj_adv_list=[] 
for sentence in morphs : 
    for word, tag in sentence : 
        if tag in ['Noun'] and ("것" not in word) and ("내" not in word)and ("나" not in word)and ("수"not in word) and("게"not in word)and("말"not in word): 
            noun_adj_adv_list.append(word) 

print(noun_adj_adv_list)

count = Counter(noun_adj_adv_list)

words = dict(count.most_common())
words

#명사가 아닌 단어 삭제
del words['심해']
del words['때']
del words['능이']

words

#word Cloud를 그리기 위한 패키지 import
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
import nltk 
from nltk.corpus import stopwords

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib 
from matplotlib import rc
rc('font', family='NanumBarunGothic')

#Word Cloud 생성
from wordcloud import WordCloud

wordcloud = WordCloud(
    font_path='malgun',    
    background_color='white',                           
    colormap = 'Accent_r',                               
    width = 800,
    height = 800
)

wordcloud_words = wordcloud.generate_from_frequencies(words)

array = wordcloud.to_array()
print(type(array))
print(array.shape)

fig = plt.figure(figsize=(10, 10))
plt.imshow(array, interpolation="bilinear")
plt.axis('off')
plt.show()
fig.savefig('business_anlytics_worldcloud.png')

data

import pandas as pd
import numpy as np

report=list(data['신고'])
y=np.array(data['분류'])

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(analyzer="word", max_features=5000)

train_data_features=vectorizer.fit_transform(report)

train_data_features

from sklearn.model_selection import train_test_split

TEST_SIZE=0.2
RANDOM_SEED=0

train_input, test_input, train_label, test_label=train_test_split(train_data_features,y,test_size=TEST_SIZE, random_state=RANDOM_SEED)


from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트 분류기에 5개의 의사결정 트리를 사용
forest=RandomForestClassifier(n_estimators=5)

# 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작
forest.fit(train_input,train_label)

# 검증 함수로 정확도 측정 train model
print("Train_Accuracy: %f" % forest.score(train_input,train_label))

# 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작
forest.fit(test_input,test_label)

# 검증 함수로 정확도 측정 train model
print("Test_Accuracy: %f" % forest.score(test_input,test_label))



pip install -U sentence-transformers

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import torch
from sentence_transformers import SentenceTransformer

answer_candidate = "C:/Users/pinkk/OneDrive/바탕 화면/jungmin data/22-2/통계학과자연어처리PBL/질문답변 데이터.xlsx"
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

df = pd.read_excel(answer_candidate)
df['embedding_vector'] = df['질문(Query)'].progress_map(lambda x : model.encode(x))
df.to_excel("train_data_embedding.xlsx", index=False)

embedding_data = torch.tensor(df['embedding_vector'].tolist()) #임베딩 데이터를 텐서화시켜서 pt파일로 저장
torch.save(embedding_data, 'embedding_data.pt')


import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

#질문 입력
sentence = input("질문 문장 : ")
sentence = sentence.replace(" ","")
print("공백 제거 문장 : ", sentence)

#문장 인코딩 후 텐서화
sentence_encode = model.encode(sentence)
sentence_tensor = torch.tensor(sentence_encode)

# 저장한 임베딩 데이터와의 코사인 유사도 측정
cos_sim = util.cos_sim(sentence_tensor, embedding_data)
print(f"가장 높은 코사인 유사도 idx : {int(np.argmax(cos_sim))}")

# 선택된 질문 출력
best_sim_idx = int(np.argmax(cos_sim))
selected_qes = df['질문(Query)'][best_sim_idx]
print(f"선택된 질문 = {selected_qes}")

# 선택된 질문 문장에 대한 인코딩
selected_qes_encode = model.encode(selected_qes)

# 유사도 점수 측정
score = np.dot(sentence_tensor, selected_qes_encode) / (np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))
print(f"선택된 질문과의 유사도 = {score}")

# 답변
answer = df['답변(Answer)'][best_sim_idx]
link = df['참고 링크(Link)'][best_sim_idx]
print(f"\n답변 : {answer}\n" + f"\n참고링크 : {link}\n")
