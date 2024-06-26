import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from datasets import load_metric
from collections import Counter
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score



class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        # TODO: Implement initialization logic
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                        num_labels=top_n_genres, problem_type="multi_label_classification")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dataset = None
        self.top_genres = []
        self.label_binarizer = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open(self.file_path, 'r') as f:
            self.dataset = json.load(f) 

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # TODO: Implement genre filtering and visualization logic
        genre_counts = {}
        for movie in self.dataset:
            if movie['genres'] is None:
                continue
            for genre in movie['genres']:
                if genre not in genre_counts:
                    genre_counts[genre] = 0
                genre_counts[genre] += 1
        
        sorted_genres = sorted(genre_counts.items(), key=lambda item: item[1], reverse=True)
        # print(sorted_genres)
        self.top_genres = [genre for genre, count in sorted_genres[:self.top_n_genres]]
        # print(self.top_genres)
        filtered_data = [movie for movie in self.dataset if movie['genres'] and 
                         movie['first_page_summary'] and any(genre in self.top_genres for genre in movie['genres'])]
        
        # mock_data = [movie for movie in self.dataset if movie['genres'] and movie['first_page_summary']]
        # print(len(mock_data))
        # print(len(self.dataset))
        # print(len(filtered_data))
        
        self.dataset = filtered_data

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # TODO: Implement dataset splitting logic
        train_size = int((1 -  test_size) * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        val_size = int(val_size * len(test_dataset))
        test_size = len(test_dataset) - val_size
        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


        # len_t = len(self.train_dataset) + len(self.test_dataset) + len(self.val_dataset)
        # print(f"train size : {len(self.train_dataset) / len_t}")
        # print(f"val size : {len(self.val_dataset) / len_t}")
        # print(f"test size : {len(self.test_dataset) / len_t}")
        


    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # TODO: Implement dataset creation logic

        return IMDbDataset(encodings=encodings, labels=labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # TODO: Implement BERT fine-tuning logic
        train_texts = []
        train_labels = []
        for movie in self.train_dataset:
            train_texts.append(movie['first_page_summary'])
            intersect_genres = list(set(movie['genres']).intersection(set(self.top_genres)))
            train_labels.append(intersect_genres)

        val_texts = []
        val_labels = []
        for movie in self.val_dataset:
            val_texts.append(movie['first_page_summary'])
            intersect_genres = list(set(movie['genres']).intersection(set(self.top_genres)))
            val_labels.append(intersect_genres)
        
        self.label_binarizer = MultiLabelBinarizer(classes=self.top_genres)
        train_labels = self.label_binarizer.fit_transform(train_labels)
        # print(train_labels[0])
        val_labels = self.label_binarizer.transform(val_labels)
        # print(train_texts[0:3])
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        # print(train_encodings)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            # logging_dir="./logs",
            # logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            # report_to=[],
            run_name="finetuning_IMDBdataset_run"
        )        

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # TODO: Implement metric computation logic
        labels = pred.label_ids
        preds = pred.predictions
        probabilities = torch.sigmoid(torch.tensor(preds))
        predictions = (probabilities > 0.5).int().numpy()
        # precision, recall, f1 = precision_recall_fscore_support()
        precision = precision_score(labels, predictions, average='samples')
        recall = recall_score(labels, predictions, average='samples')
        f1 = f1_score(labels, predictions, average='samples')
        return {
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
                }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # TODO: Implement model evaluation logic
        test_texts = []
        test_labels = []

        for movie in self.test_dataset:
            test_texts.append(movie['first_page_summary'])
            intersect_genres = list(set(movie['genres']).intersection(set(self.top_genres)))
            test_labels.append(intersect_genres)

        test_labels = self.label_binarizer.transform(test_labels)
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=512)
        test_dataset = self.create_dataset(test_encodings, test_labels)

        #######
        trainer = Trainer(model=self.model, compute_metrics=self.compute_metrics)
        metrics = trainer.evaluate(test_dataset)
        print(metrics)



    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)

class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # TODO: Implement item retrieval logic
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)