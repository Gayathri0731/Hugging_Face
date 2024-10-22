from transformers import pipeline #type:ignore

from transformers import  AutoTokenizer, AutoModelForSequenceClassification #type:ignore

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis',model=model, tokenizer=tokenizer)
res = classifier('Hii, i feel not good today')

print('Hii, i feel not good today : ')
print(res)