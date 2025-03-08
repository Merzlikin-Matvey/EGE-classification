import pandas as pd
from src.model import TaskClassifier
import torch

model = TaskClassifier()
model.train('dataset/tasks.csv')
print(model.predict('Задача по геометрии'))
print(model.predict('Задача по алгебре'))