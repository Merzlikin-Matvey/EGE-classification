import transformers
import pandas as pd
import datasets
import torch
import datetime


class TaskClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=19):
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class

    def preprocess(self, dataset_path):
        dataset = datasets.Dataset.from_pandas(pd.read_csv(dataset_path))
        dataset = dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, padding=True), batched=True)
        return dataset

    def train(self, dataset_path):
        print("Training model...")
        dataset = self.preprocess(dataset_path)
        dataset = dataset.map(lambda x: {'labels': x['label']}, batched=True)
        train_test_split = dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                logging_dir='logs',
                logging_steps=10,
                num_train_epochs=1,
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                save_total_limit=2
            )
        )
        trainer.train()
        self.model = trainer.model

    def save(self):
        name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path = "models/" + name
        self.model.save_pretrained(path)