# Illuin Technology Assessment - Adam FERHOUT

## Introduction

**FRENCH DISCLAIMER** : Le code de cet exercice à été rédigé et commenté en anglais intégralement de manière à le rendre lisible pour n'importe quel examinateur. 

**USAGE OF AI** : Since no specific rules regarding AI were specified, AI was used in this project on a case-by-case basis. I wrote the majority of the code myself without using AI. However, an LLM (Claude) was used for finishing tasks, such as formatting graphs with matplotlib, drafting explorations of new avenues, or interacting with the documentation for certain libraries.

This project predicts the tags of a Codeforces exercise. The tags are restrained to "math", "graphs", "strings", "number theory", "trees", "geometry", "games" and "probabilities", and the prediction is made from an exercise's description and source code.

## Setup

Made with python 3.12

Setup code env :

python -m venv .venv
.venv\Scripts\activate 
pip install -r requirements.txt

Unzip the dataset in "data/code_classification_dataset" and create the env from env example

## Train the model

Run every command from the project root.

**This should always be run first to train** the model. Splits the dataset into train (80%) and test (20%) sets, saves both in the data folder as parquet files, and saves the model with joblib in the "models" folder.

*Bash :* 
python -m src.train

## Run predictions

You can either run specific predictions on one sample, or on the whole test set.

**To Predict** the tags of one test exercise. 
python -m src.main predict <src_uid>
-> Return predicted tags and Ground Truth

Without an id, it lists some test ids.
python -m src.main predict
-> Returns some test ids that can be used to run a prediction


**Evaluate** the model on the test set (precision, recall and F1 per tag) using the sklearn classificaiton report.
This also returns the metrics per tag.

python -m src.main evaluate


## Exploration

"src/exploration.ipynb" contains the data exploration and the model comparison. The zero-shot LLM part needs a Groq API key in ".env" ("GROQ_API_KEY=...").
