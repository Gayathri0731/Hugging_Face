from transformers import pipeline #type:ignore

classifier = pipeline('zero-shot-classification')

res = classifier(
    'I have completed this course',
    candidate_labels=['education','politics','business']
)
print(res)

# from transformers import pipeline

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# candidate_labels = ['education', 'politics', 'business']

# result = classifier("there is a great loss in our company", candidate_labels)

# print(result)
