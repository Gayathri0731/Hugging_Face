# from transformers import pipeline; 
# print(pipeline('sentiment-analysis')('we hate you'))

#alias

from transformers import pipeline; # type: ignore
classifier = pipeline('sentiment-analysis')
res = classifier('Hii, i feel not good today')
res1 = classifier("I love Python")
print('Hii, i feel not good today : ',res)
print("I love Python : ",res1)