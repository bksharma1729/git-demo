# Problem Statement
'''Customer acquisition is expensive("Customer acquisition is 
expensive" means the total cost to get a new customer (CAC) involves 
significant spending on marketing, sales salaries, ads, content, and 
tools, often far exceeding the cost to keep an existing customer, highlighting 
a business challenge where high costs can reduce profit unless balanced by
 high customer lifetime value (LTV) or efficient strategies like retention). 
 Businesses want 
to predict which customers are likely to leave (churn) 
so they can take preventive action.

Objective
Build a machine learning model to predict customer
churn and identify the key factors influencing churn.
'''

'''Business Impact (Interview Angle):-
Helps reduce revenue loss
Enables targeted retention campaigns
Improves customer lifetime value'''
#Importing Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Loading Dataset
data=pd.read_csv("E:\\6th Semester\\PROJECT_ML\\telecom_customer_churn.csv")
#Display first few rows
print("First 5 rows of data:")
print(data.head())
#Dataset Shape
print("\nShape of Data:")
print(data.shape)
# Missing values
print("\nMissing values (column-wise):")
print(data.isnull().sum())
#Data Types
print("\nData Types:")
print(data.dtypes)
#Statistical summary
print("\nStatistical Summary:")
print(data.describe())
