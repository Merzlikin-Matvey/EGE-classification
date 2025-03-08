import pandas as pd
import torch

from src.model import TaskClassifier

model = TaskClassifier().load("models/2025-03-08-22-39-13")
print(model.predict("Какова вероятность того что последние две цифры случайного телефонного номера различны?"))