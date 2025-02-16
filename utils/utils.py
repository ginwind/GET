import re
import math
import torch
import numpy as np
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.stats import spearmanr
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            relations,
            tokenizer,
            epoch: int = 3,
            batch_size: int = 64,
            warm_up_rate: float = 0.1, 
            lr: float=1e-5,
            lr_min: float=1e-6,
            device: torch.device=torch.device("cpu"),
            model: nn.Module=None,
            words=None
        ):
        self.words=words
        self.relations = relations
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.warm_up_epoch = int(epoch * warm_up_rate)
        self.batch_size = batch_size
        self.lr = lr
        self.lr_min = lr_min
        self.device = device
        self.model = model.to(device=device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmupCosineAnnealingLR)

    def warmupCosineAnnealingLR(self, cur_iter):
        if cur_iter < self.warm_up_epoch:
            return ((self.lr - self.lr_min) * (cur_iter / self.warm_up_epoch) + self.lr_min) / self.lr
        else:
            return (self.lr_min + 0.5 * (self.lr - self.lr_min) * (
                1 + math.cos((cur_iter - self.warm_up_epoch) / (self.epoch - self.warm_up_epoch) * math.pi)
            )) / self.lr

    def train_le(self, train_loader, test_loader, val_loader):
        val_f1, test_f1 = [], []
        for i in range(0, self.epoch):
            self.model.train()
            prediction, labels = [], []
            for data in tqdm(train_loader):
                data = {key: value.to(self.device) for key, value in data.items()}
                outputs = self.model(**data)

                l, logits = outputs["loss"], outputs["logits"]

                print(l)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                prediction.extend(logits.flatten().cpu().tolist())
                labels.extend(data["labels"].cpu().tolist())
            self.scheduler.step()
            sp_train = spearmanr(labels, prediction)
            print(f"epoch{i}, training_spearman: {sp_train}")
            if val_loader is not None:
                val_f1.append(self.evaluate_le(val_loader, "val"))
            test_f1.append(self.evaluate_le(test_loader))
        if len(val_f1) > 0:
            print(f"best spearman: {test_f1[np.argmax(val_f1)]}")
        else:
            print(f"last spearman: {test_f1[-1]}")

    def evaluate_le(self, dataloader, data_split="test"):
        self.model.eval()
        prediction, labels = [], []
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = {key: value.to(self.device) for key, value in data.items()}
                outputs = self.model(**data)

                prediction.extend(outputs["logits"].flatten().cpu().tolist())
                labels.extend(data["labels"].cpu().tolist())
            spear = spearmanr(labels, prediction)
            print(f"{data_split} spearman:{spear}")
        return spear[0]
    
    def train(self, train_loader, test_loader, val_loader):
        val_f1, test_f1 = [], []
        for i in range(0, self.epoch):
            self.model.train()
            prediction, labels = [], []
            for data in tqdm(train_loader):
                data = {key: value.to(self.device) for key, value in data.items()}
                outputs = self.model(**data)

                l, logits = outputs["loss"], outputs["logits"]

                print(l)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                prediction.extend(torch.argmax(logits, dim=1).cpu().tolist())
                labels.extend(data["labels"].cpu().tolist())
            self.scheduler.step()
            print(f"epoch{i}, training weighted f1:{f1_score(labels, prediction, average='weighted')}, \
                  weighted precision: {precision_score(labels, prediction, average='weighted')}, \
                  weighted recall: {recall_score(labels, prediction, average='weighted')}")
            if val_loader is not None:
                val_f1.append(self.evaluate(val_loader, "val"))
            test_f1.append(self.evaluate(test_loader))
        if len(val_f1) > 0:
            print(f"best weighted f1: {test_f1[np.argmax(val_f1)]}")
        else:
            print(f"last weighted f1: {test_f1[-1]}")
        
    def evaluate(self, dataloader, data_split="test"):
        self.model.eval()
        prediction, labels = [], []
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = {key: value.to(self.device) for key, value in data.items()}
                outputs = self.model(**data)

                prediction.extend(torch.argmax(outputs["logits"], dim=1).cpu().tolist())
                labels.extend(data["labels"].cpu().tolist())
            f1 = f1_score(labels, prediction, average='weighted')
            print(f"{data_split} weighted f1:{f1}, \
                    weighted precision: {precision_score(labels, prediction, average='weighted')}, \
                weighted recall: {recall_score(labels, prediction, average='weighted')}")
            if data_split == "test":
                print(classification_report(labels, prediction, target_names=self.relations))
                print(confusion_matrix(labels, prediction))
        return f1
    
    def train_generation(self, train_loader, test_loader, val_loader):
        self.model.train()
        val_f1, test_f1 = [], []
        for i in range(0, self.epoch):
            for data in tqdm(train_loader):
                data = {key: value.to(self.device) for key, value in data.items()}
                outputs = self.model(**data)

                l = outputs["loss"]

                print(l)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
            self.scheduler.step()
            if val_loader is not None:
                val_f1.append(self.evaluate_generation(val_loader, "val"))
            test_f1.append(self.evaluate_generation(test_loader))
        if len(val_f1) > 0:
            print(f"best weighted f1: {test_f1[np.argmax(val_f1)]}")
        else:
            print(f"last weighted f1: {test_f1[-1]}")

    def evaluate_generation(self, dataloader, data_split="test"):
        def filter_words(string):
            words = re.findall(r'\b\w+\b', string)
            
            return [word.strip() for word in words if word in self.relations]

        def str_classification(new_str):
            result = []
            for i in range(0, len(self.relations)):
                if self.relations[i] == new_str:
                    result.append(i)
            if len(result) != 1:
                return -1
            else:
                return result[0]
        
        self.model.eval()
        prediction, labels = [], []
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = {key: value.to(self.device) for key, value in data.items()}
                outputs = self.model.generate(max_new_tokens=128, temperature=0.5, **data)

                responses = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                )

                for response in responses:
                    response = response.split("Answers:")[1]
                    response = filter_words(response)
                    print(response, len(response), int(data["labels"].shape[1]))
                    response_labels = []
                    for k in range(0, len(response)):
                        response_labels.append(str_classification(response[k]))
                    if len(response_labels) < int(data["labels"].shape[1]):
                        response_labels = response_labels + [-1] * (int(data["labels"].shape[1]) - len(response_labels))
                    elif len(response_labels) > int(data["labels"].shape[1]):
                        response_labels = response_labels[:int(data["labels"].shape[1])]
                    prediction.extend(response_labels)
                labels.extend(data["labels"].flatten().cpu().tolist())

            f1 = f1_score(labels, prediction, average='weighted')
            print(f"{data_split} weighted f1:{f1}, \
                    weighted precision: {precision_score(labels, prediction, average='weighted')}, \
                weighted recall: {recall_score(labels, prediction, average='weighted')}")
            if data_split == "test":
                print(classification_report(labels, prediction, target_names=self.relations))
                print(confusion_matrix(labels, prediction))
        return f1
            