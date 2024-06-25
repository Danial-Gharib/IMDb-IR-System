from BertFinetuner_mask import BERTFinetuner


# Instantiate the class
bert_finetuner = BERTFinetuner('IMDB_crawled_standard.json', top_n_genres=5)

# Load the dataset
bert_finetuner.load_dataset()
# print(bert_finetuner.dataset[0].keys())
print("dataset loaded successfully")
# Preprocess genre distribution
bert_finetuner.preprocess_genre_distribution()
print("genre distribution preprocessed successfully")
# Split the dataset
bert_finetuner.split_dataset(test_size=0.2, val_size=0.5)
print("data split successfully")
# Fine-tune BERT model
bert_finetuner.fine_tune_bert()
print("bert model finetuned successfully")
# Compute metrics
bert_finetuner.evaluate_model()
print("model evaluated successfully")
# Save the model (optional)
bert_finetuner.save_model('Movie_Genre_Classifier')
print("bert model saved successfully")