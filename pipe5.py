from transformers import pipeline #type:ignore
from transformers import  AutoTokenizer, AutoModelForSequenceClassification #type:ignore

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


sequence = "Hii, I am glad to meet you"
res = tokenizer(sequence)
print("Result : ",res)

tokens = tokenizer.tokenize(sequence)
print("\n tokens : ",tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print("\n ids : ",ids)

decode = tokenizer.decode(ids)
print("\n Decoded String : ",decode)