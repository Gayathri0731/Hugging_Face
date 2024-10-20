from transformers import pipeline  # type: ignore

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "in this course we will teach you how to",
    max_length=30,  # Corrected typo here
    num_return_sequences=5,
)
print(res)
