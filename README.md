# Predicting Credit Score (WIP) - 82% accuracy

## Short Description
Looking at a dataset taken from kaggle (https://www.kaggle.com/datasets/parisrohan/credit-score-classification), can I build a ML model to accurately predict someones credit score.

## Current Status
- Finished cleaning up the data by filling in missing data. 
  - Already I can see areas in which I have approximated missing data in maybe a suboptimal way and therefore there is potential to alter how I deal with this missing data and consequently this could improve (or make worse) my ML models but we shall see.
- Added a Neural Network (NN) using TensorFlow and have managed to achieve an accuracy of 69%.
- By using SMOTE (Synthetic Minority Oversampling TEchnique) I corrected for the imbalanced data set and improved the accuracy of the NN to 74%.
- The plan is to now look into the models provided by Scikit-Learn and see if I can improve this score.
  - Implemented Logistic Regression model although performance was 64% without SMOTE and 67% with SMOTE so performed worse than the NN.
  - Implemented K-Nearest Neighbours (with SMOTE) with accuracy 82%.

## What I've learned
- In hindsight, if I had spent a bit more time observing why a large proportion of the numerical columns were not in fact numerical I could have saved myself some time by writing a more generic function to correct this.
- The display() function in jupyter notebooks massively helped in trying to debug the function written to fill in missing data from "Credit_History_Age".
- The ffill() and bfill() functions are very useful in filling in missing data which is either the same for a number of rows or follows some sequential pattern like in the case of this notebook.
- Another in hindsight, I think splitting the notebook into separate categories (e.g preparing the data, further EDA, ML training) might speed up the overall process and improve readability of the code
