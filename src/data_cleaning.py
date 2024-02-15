import pandas as pd
import numpy as np

def main():
  df = pd.read_csv("data/train.csv", low_memory=False)

  # Drop columns irrelevant to ML model
  df.drop(['ID', 'Month', 'Name', 'SSN', 'Monthly_Inhand_Salary', 'Credit_Mix'], axis=1, inplace=True)

  # Preparing columns
  df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].apply(lambda credit_limit: np.nan if credit_limit=='_' else credit_limit)

  # Correcting outliers
  columns_with_underscores = ['Age', 'Annual_Income', 'Num_Bank_Accounts', 
                              'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                              'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                              'Num_Credit_Inquiries', 'Outstanding_Debt',
                              'Amount_invested_monthly', 'Monthly_Balance']

  min_value = {'Age': 0,
              'Annual_Income': 0,
              'Num_Bank_Accounts': 0,
              'Num_Credit_Card': 0, 
              'Interest_Rate': 0, 
              'Num_of_Loan': 0, 
              'Num_of_Delayed_Payment': 0, 
              'Changed_Credit_Limit': -np.inf, 
              'Num_Credit_Inquiries': 0, 
              'Outstanding_Debt': -np.inf, 
              'Amount_invested_monthly': 0, 
              'Monthly_Balance': 0}
  max_value = {'Age': 60, 
              'Annual_Income': 200000, 
              'Num_Bank_Accounts': 20, 
              'Num_Credit_Card': 15, 
              'Interest_Rate': 40, 
              'Num_of_Loan': 15, 
              'Num_of_Delayed_Payment': 30, 
              'Changed_Credit_Limit': np.inf, 
              'Num_Credit_Inquiries': 20, 
              'Outstanding_Debt': np.inf, 
              'Amount_invested_monthly': 2000, 
              'Monthly_Balance': np.inf}
  
  for col in columns_with_underscores:
    df[col] = df[col].apply(lambda s: float(s.replace("_","")) if isinstance(s, str) else s)
    # Remove outliers and replace with customer mean
    df[col] = df[col].apply(lambda val: np.nan if (val < min_value[col] or val > max_value[col]) else val)
    customer_val = df.groupby('Customer_ID')[col].mean()
    df[col] = df.apply(lambda x: fill_missing_value(customer_val, x['Customer_ID'], x[col]), axis=1)

  # Convert credit history to number of months
  df['Credit_History_Age'] = df['Credit_History_Age'].apply(lambda s : 12*float(s.split()[0]) + float(s.split()[3]) if s is not np.nan else np.nan)
  df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(fill_missing_credit_history_age)
  # Convert credit score to numbers for ML model
  df['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

  # Categorical columns
  # - Fix missing values
  df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(lambda x: x.replace("_______", np.nan))
  df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(fill_occupation)
  df['Type_of_Loan'].replace([np.NaN], 'Not Specified', inplace=True)
  df['Type_of_Loan'] = df['Type_of_Loan'].apply(lambda x: x.lower().replace('and ', '').replace(', ', ',').strip())
  df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].apply(lambda x : "Yes" if x=="NM" else x)
  df = df.drop(df[df['Payment_Behaviour'] == '!@9#%8'].index)
  
  # - Create dummy variables
  df = pd.concat([df, pd.get_dummies(df['Occupation'], prefix='Occupation', drop_first=True),
                  df['Type_of_Loan'].str.get_dummies(sep=',').add_prefix('type_of_loan_'),
                  pd.get_dummies(df['Payment_of_Min_Amount'], prefix='Payment_of_Min_Amount', drop_first=True),
                  pd.get_dummies(df['Payment_Behaviour'], prefix='Payment_Behaviour', drop_first=True)],axis=1)

  # Drop columns after creating dummy variables
  df.drop(['Occupation', 'Type_of_Loan', 'Payment_of_Min_Amount', 'Payment_Behaviour'], axis=1, inplace=True)
  # Drop customer_ID now no longer needed
  df.drop('Customer_ID', axis=1, inplace=True)
  
  df.to_csv("data/ML_ready.csv")

  print(df.isnull().sum().apply(lambda x: 100*(x/len(df))))
  
def fill_missing_value(customer_val, customer_ID, val):
    return customer_val[customer_ID] if np.isnan(val) else val
  
def fill_missing_credit_history_age(series):
    filled_forward = series.ffill() + series.groupby(series.notna().cumsum()).cumcount()
    filled_backward = filled_forward.bfill() - filled_forward.groupby(filled_forward.notna().cumsum()).cumcount(ascending=False)
    return filled_backward
  
def fill_occupation(series):
  filled_forward = series.ffill()
  return filled_forward.bfill()

if __name__ == '__main__':
  main()