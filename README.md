# gpt-api-evaluation
This repository contains the implementation of a text classifier and a large language model for decision making, using the Simple Transformers library with a BERT model and the GPT-API respectively. The model is trained and evaluated on a dataset containing the "Author Keywords" and associated "Decisions" which are categorised as 'Included', 'Rejected', or 'Not Sure'.

## Overview
The code reads data from a CSV file and selects the relevant columns.
A BERT model is trained for each column of the dataset, and evaluation metrics (MCC, F1 score, Accuracy, and Eval loss) are calculated and plotted after each epoch.
The GPT models cannot be trained but can be prompted by giving context.
In addition, confusion matrices are generated using seaborn to visually understand the performance of the model's predictions.
## Requirements
pandas
sklearn
simpletransformers
matplotlib
numpy
seaborn
torch
cuda
### !! IMPORTANT !! 
The BERT model cannot be run without CUDA!
## Steps Involved
Import necessary libraries.
Generate and plot the evaluation metrics.
## Notebooks
BERT.ipynb: Contains the main code for reading the dataset, preparing it, training the BERT model, and evaluating the model's performance.

GPT3-EVALUATION.ipynb: Code for generating and plotting the confusion matrix using the GPT-3 Dataset.

GPT4-EVALUATION.ipynb: Code for generating and plotting the confusion matrix using the GPT-4 Dataset.


## Future Improvements
The project can be enhanced by fine-tuning the model using hyperparameter optimization and by using a larger, more diverse dataset. We can also explore different models for better performance.

## Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.