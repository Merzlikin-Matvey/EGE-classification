from src.huggingface import HuggingFace
from src.model import TaskClassifier

model = TaskClassifier()

while True:
    print(model.predict(input("Введите текст: ")))