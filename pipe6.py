from transformers import pipeline #type:ignore
from transformers import  AutoTokenizer, AutoModelForSequenceClassification #type:ignore
import torch #type:ignore
import torch.nn.functional as F #type:ignore


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis',model=model, tokenizer=tokenizer)

X_train= ["Hii, I am glad to meet you","I feel bad"]

res= classifier(X_train)
print(res)

batch = tokenizer(X_train, padding= True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    Predictions = F.softmax(outputs.logits,dim=1)
    print(Predictions)
    Labels = torch.argmax(Predictions,dim=1)
    print(Labels)