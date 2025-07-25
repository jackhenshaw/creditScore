# Predicting Credit Score Bracket - 87.6% accuracy

**Disclaimer:** This project was created early in my learning journey and demonstrates foundational machine learning concepts. While basic, it showcases my initial exploration of supervised learning techniques.

## Short Description
Took dataset from kaggle (https://www.kaggle.com/datasets/parisrohan/credit-score-classification): cleaned the data, performed feature engineering on missing values, performed EDA looking at correlations, followed by testing a whole range of ML models.

## Status
- Finished data cleaning by filling in missing data through feature engineering. 
  - Already I can see areas in which I have approximated missing data in maybe a suboptimal way and therefore there is potential to alter how I deal with this missing data and consequently this could improve (or make worse) my ML models.
- Looked at correlations between variables and noted which columns were closely matched with having a "good" credit score.
- Corrected imbalanced dataset using SMOTE (Synthetic Minority Oversampling TEchnique) greatly improving accuracy of my models.
- Tested a whole range of ML models:
  - Neural Network using Tensorflow
  - Scikit Learn:
    - Logistic Regression
    - K-Nearest Neighbours
    - LinearSVC
    - Decision Tree
    - Random Forest Classifier
    - HistGradientBossting Classifier
- Fine tuned the hyperparameters using a gridsearchCV.  

## Results
Hypertuning the parameters of the random forest model results in an accuracy of 87.6% and an f1-score of 0.875. <br>
Accuracy and f1-scores for all (untuned) models are shown below:

| Method                                 | Accuracy |
|----------------------------------------|----------|
| Random Forest (SMOTE)                  | 87.18%   | 
| KNN (SMOTE)                            | 84.81%   | 
| HistGradientBoostingClassifier (SMOTE) | 81.11%   | 
| Decision Tree (SMOTE)                  | 78.34%   | 
| Neural Network (SMOTE)                 | 73.95%   | 
| Neural Network                         | 69.06%   | 
| Logistic Regression (SMOTE)            | 66.92%   | 
| LinearSVC (SMOTE)                      | 66.61%   | 
| Logistic Regression                    | 64.29%   | 

| Method                                 | f1-score |
|----------------------------------------|----------|
| Random Forest (SMOTE)                  | 87.08%   |  
| KNN (SMOTE)                            | 84.54%   |  
| HistGradientBoostingClassifier (SMOTE) | 81.05%   |  
| Decision Tree (SMOTE)                  | 78.29%   | 
| Neural Network (SMOTE)                 | 73.61%   |  
| Neural Network                         | 68.68%   |  
| Logistic Regression (SMOTE)            | 66.52%   |  
| LinearSVC (SMOTE)                      | 65.74%   |  
| Logistic Regression                    | 63.46%   |

## What I've learned
- In hindsight, if I had spent a bit more time observing why a large proportion of the numerical columns were not in fact numerical I could have saved myself some time by writing a more generic function to correct this. -> This is now done in the source file data_cleaning.py
- The display() function in jupyter notebooks massively helped in trying to debug the function written to fill in missing data from "Credit_History_Age".
- The ffill() and bfill() functions are very useful in filling in missing data which is either the same for a number of rows or follows some sequential pattern like in the case of this notebook.
- Another in hindsight, I think splitting the notebook into separate categories (e.g preparing the data, further EDA, ML training) might speed up the overall process and improve readability of the code -> I did implement this in the end.
- Methods to correct an imbalanced dataset are very useful in improving the performance of ML models. In this case I used Synthetic Minority Oversampling TEchnique (SMOTE).
