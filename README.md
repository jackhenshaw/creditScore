# Credit Score Classification: An Introductory Machine Learning Project - 87.6% Accuracy

**Context:** This project represents an early exploration into supervised learning and classification techniques, specifically applied to credit scoring. It showcases my foundational understanding of these algorithms and the core principles of building a predictive model. Looking back with more industry experience, this project reinforced key lessons in robust model evaluation and the critical importance of a clear business objective in predictive analytics.

---

## Project Objective

The primary objective of this project was to develop a binary classification model capable of predicting the creditworthiness of loan applicants, categorising them as 'good' (low risk) or 'bad' (high risk). This is a foundational problem in financial risk management, directly impacting lending decisions and mitigating potential losses for financial institutions.

**Dataset:** The analysis utilises the German Credit Data, a public dataset detailing various attributes of loan applicants and their credit status. [Link to Kaggle Dataset](https://www.kaggle.com/datasets/santoshgond/geman-credit-data/data)

---

## Methodology

This project followed a typical supervised machine learning workflow:

1.  **Data Preprocessing:**
    * Conducted initial exploratory data analysis (EDA) to understand data characteristics.
    * Performed necessary encoding to transform categorical features into a numerical format suitable for machine learning algorithms.
2.  **Model Experimentation & Selection:**
    * Trained and evaluated a range of classification algorithms, including Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbours (KNN), Random Forest, AdaBoost, and Naive Bayes.
    * Initial model performance was assessed using accuracy score.
3.  **Model Optimisation:**
    * The best-performing model (Random Forest) underwent hyperparameter tuning to further optimise its performance.
    * Final model evaluation involved assessing accuracy and analysing the confusion matrix to understand true positives, false positives, true negatives, and false negatives – crucial for risk assessment.

---

## Key Findings & Business Reflection

* **Model Performance:** The Random Forest classifier emerged as the best-performing model, achieving an accuracy of 0.88.
* **Understanding Classification Challenges (Business Context):** Crucially, the confusion matrix revealed that while the model performed well in identifying 'bad' loan applicants (minimising false negatives, which is vital for preventing financial losses), it struggled more with correctly classifying 'good' applicants (leading to higher false positives). This highlights a common challenge in credit scoring:
    * **Asymmetry of Costs:** In a real-world lending scenario, misclassifying a 'bad' applicant as 'good' (a false negative) can lead to significant financial loss. Conversely, misclassifying a 'good' applicant as 'bad' (a false positive) means a missed revenue opportunity.
    * **Beyond Accuracy:** This project reinforced the importance of selecting evaluation metrics tailored to the business problem. While accuracy is a general metric, metrics like precision, recall, and F1-score (especially for the 'bad' class) would be more appropriate for a comprehensive credit risk assessment, allowing a business to tune the model's sensitivity based on its risk appetite. My industry experience has further solidified this understanding, showing how classification models are carefully balanced against specific business key performance indicators (KPIs) and risk tolerance.

* **Model Deployment:** The trained Random Forest model has been saved as a pickle file, demonstrating readiness for potential integration into a larger system for inference.
