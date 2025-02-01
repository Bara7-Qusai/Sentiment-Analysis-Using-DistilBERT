# Sentiment Analysis Using DistilBERT

## Overview
This project demonstrates how to fine-tune a `DistilBERT` model for sentiment analysis using `Hugging Face Transformers` and `Datasets` libraries. It utilizes the IMDb reviews dataset, which consists of positive and negative sentiment labels.

## Installation
Ensure you have the required dependencies installed before running the code. If you have an older version of `datasets`, install the required version:

```bash
pip install -q "datasets==2.15.0"
```

## Model
The project uses the `distilbert-base-uncased` model with a sequence classification head. The model is configured as follows:
- `num_labels=2` (Binary classification: Positive/Negative)
- `id2label` and `label2id` mappings to convert predictions into readable labels
- Freezing the base model parameters to prevent unnecessary updates

## Data Preparation
- The IMDb dataset is loaded using `datasets.load_dataset("imdb")`.
- The dataset is shuffled and reduced to 500 samples per split (`train` and `test`) for faster execution.
- Tokenization is performed using `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
- The tokenized data is stored in `tokenized_ds` for training and evaluation.

## Training & Evaluation
The `Trainer` from `Hugging Face` is used to manage model training and evaluation:
- Learning rate: `2e-3`
- Batch size: `4` (adjustable based on memory availability)
- Number of epochs: `1`
- Weight decay: `0.01`
- Evaluation and saving strategy: `epoch`
- `DataCollatorWithPadding` is used to handle padding in input sequences
- Accuracy is used as the evaluation metric

### Start Training
```python
trainer.train()
```

### Run Evaluation
```python
trainer.evaluate()
```

## Prediction & Analysis
- The model predicts sentiment labels for the test dataset.
- Predictions are stored in a `Pandas` dataframe.
- Texts are cleaned by replacing `<br />` tags with spaces.
- Predictions are added as a new column in the dataframe.

## Example Output
```python
# Display the first two rows of the dataframe
df.head(2)
```
This displays the first two rows of the test dataset with actual and predicted labels.

## References
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [IMDb Dataset on Hugging Face](https://huggingface.co/datasets/imdb)



