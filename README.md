# Ensemble PhoBERT & FastText in Vietnamese Sentiment Analysis task

## Introduction

## Dataset

## Model

## Result

### Evaluation on Test Set

| Model | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| (1) PhoBERT (base) + FeedForward | **0.92502** | **0.92988** | **0.92348** |
| (2) PhoBERT (large) + FeedForward | 0.91447 | 0.90935 | 0.88475 |
| (3) PhoBERT (base) + LSTM | 0.92399 | 0.92893 | 0.92259 |
| (4) PhoBERT (large) + LSTM | 0.91062 | 0.90556 | 0.88104 |
| (5) FastText + LSTM | 0.84022 | 0.86323 | 0.84127 |
| (6) FastText + SVM  | 0.84825 | 0.86639 | 0.85023 |

### **Emsemble** evaluation on Test Set

| Model | Ratio | Precision | Recall | F1-score |
| ----- | ----- | --------- | ------ | -------- |
| (2) + (6) | 0.5 | **0.89417** | **0.91124** | **0.88877** |
| (2) + (4) | 0.7 | 0.91587 | 0.91093 | 0.88627 |
| (2) + (5) | 0.8 | 0.91521 | 0.91030 | 0.88565 |
| (4) + (6) | 0.2 | 0.89082 | 0.90556 | 0.88562 |
| (4) + (5) | 0.7 | 0.91145 | 0.90651 | 0.88195 |
| (5) + (6) | 0.4 | 0.85532 | 0.87208 | 0.85340 |


### Evaluation on Test set with **class weights**

| Model | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| (1) PhoBERT (base) + FeedForward | **0.92867** | **0.92672** | **0.92751** |
| (2) PhoBERT (large) + FeedForward | 0.90756 | 0.9024 | 0.87796 |
| (3) PhoBERT (base) + LSTM | 0.92489 | 0.92356 | 0.92407 |
| (4) PhoBERT (large) + LSTM | 0.90965 | 0.90461 | 0.8801 |
| (5) FastText + LSTM | 0.85727 | 0.81207 | 0.83015 |
| (6) FastText + SVM  | 0.85376 | 0.86229 | 0.85561 |

### **Ensemble** Evaluation on Test set with **class weights**

| Model | Ratio | Precision | Recall | F1-score |
| ----- | ----- | --------- | ------ | -------- |
| (1) + (4) | 0.8 | **0.92845** | **0.92956** | **0.92889** |
| (1) + (2) | 0.9 | 0.92899 | 0.92798 | 0.92837 |
| (1) + (6) | 0.5 | 0.92932 | 0.92830 | 0.92830 |
| (1) + (5) | 0.9 | 0.92943 | 0.92672 | 0.92783 |
| (3) + (4) | 0.8 | 0.92507 | 0.92704 | 0.92584 |
| (3) + (6) | 0.8 | 0.92545 | 0.92451 | 0.92484 |
| (3) + (5) | 0.6 | 0.92654 | 0.92356 | 0.92474 |