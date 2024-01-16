# Predicting Credit Score Bracket (WIP) - 87% accuracy

## Results
Further feature engineering has increased the accuracy up to 87.07% using a Random Forest model and the SMOTE for oversampling the imbalanced dataset.

| Method                                 |  Accuracy | 
|----------------------------------------|-----------|
| Random Forest (SMOTE)                  |   85.99%  | 
| KNN (SMOTE)                            |   82.11%  | 
| HistGradientBoostingClassifier (SMOTE) |   80.96%  |
| Decision Tree (SMOTE)                  |   77.41%  | 
| SVM (SMOTE)                            |   74.55%  | 
| Neural Network (SMOTE)                 |   74.29%  | 
| Neural Network                         |   68.74%  | 
| Logistic Regression (SMOTE)            |   67.01%  | 
| Logistic Regression                    |   64.49%  |

## Short Description
Took dataset from kaggle (https://www.kaggle.com/datasets/parisrohan/credit-score-classification), cleaned up the data, performed EDA followed by testing a whole range of ML models.

## Current Status
- Finished cleaning up the data by filling in missing data. 
  - Already I can see areas in which I have approximated missing data in maybe a suboptimal way and therefore there is potential to alter how I deal with this missing data and consequently this could improve (or make worse) my ML models but we shall see.
- Added a Neural Network (NN) using TensorFlow and have managed to achieve an accuracy of 69%.
- By using SMOTE (Synthetic Minority Oversampling TEchnique) I corrected for the imbalanced data set and improved the accuracy of the NN to 74%.
- Looked into a range of models provided by Scikit-Learn to try and improve this score:
  - Implemented Logistic Regression model although performance was 64% without SMOTE and 67% with SMOTE so performed worse than the NN.
  - Implemented K-Nearest Neighbours (with SMOTE) with 82% accuracy.
  - Implemented SVM model (with SMOTE) with 74% accuracy.
  - Implemented both Decision Tree (78% accuracy) and Random Forest Classifiers (86% accuracy).
  - Implemented a HistGradientBoosting Classifier (81% accuracy) after reading it might perform better than the Random Forest. Sadly it did not, I think for multiclass classifications this is not always true.
- Can I use GridSearch to fine tune the Random Forest Classifier? -> First let's try feature engineering to improve performance.

## What I've learned
- In hindsight, if I had spent a bit more time observing why a large proportion of the numerical columns were not in fact numerical I could have saved myself some time by writing a more generic function to correct this.
- The display() function in jupyter notebooks massively helped in trying to debug the function written to fill in missing data from "Credit_History_Age".
- The ffill() and bfill() functions are very useful in filling in missing data which is either the same for a number of rows or follows some sequential pattern like in the case of this notebook.
- Another in hindsight, I think splitting the notebook into separate categories (e.g preparing the data, further EDA, ML training) might speed up the overall process and improve readability of the code -> Now implemented this.
- Methods to correct an imbalanced dataset are very useful in improving the performance of ML models. In this case I used Synthetic Minority Oversampling TEchnique (SMOTE).
