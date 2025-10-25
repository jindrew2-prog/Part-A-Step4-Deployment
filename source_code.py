# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 06:23:42 2025

@author: jindr
"""

import os
#from pathlib2 import Path
from zipfile import ZipFile
import time
import glob

import pandas as pd
import numpy as np
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score

# download the files

# <note: make them all relative, absolute path is not accepted>
zip_path = 'data_compressed'
#base_path = 'OneDrive - University of Canberra (student)'
csv_base_path = 'csv_files'

#!mkdir -p {csv_base_path}
os.makedirs(csv_base_path, exist_ok=True)

# How many zip files do we have? write a code to answer it.
zip_files = [f for f in os.listdir(zip_path) if f.lower().endswith('.zip')]
print(f"Number of zip files found: {len(zip_files)}")

def zip2csv(zipFile_name , file_path):
    """
    Extract csv from zip files
    zipFile_name: name of the zip file
    file_path : name of the folder to store csv
    """

    try:
        with ZipFile(zipFile_name, 'r') as z: 
            print(f'Extracting {os.path.basename(zipFile_name)} ') 
            z.extractall(path=file_path) 
    except:
        print(f'zip2csv failed for {os.path.basename(zipFile_name)}')

for file in zip_files:
    full_zip_path = os.path.join(zip_path, file)
    #print(full_zip_path)
    zip2csv(full_zip_path, csv_base_path)

print("Files Extracted")

# How many csv files have we extracted? write a code to answer it.
csv_files = [f for f in os.listdir(csv_base_path) if f.lower().endswith('.csv')]

num_csv_files = len(csv_files)
print(f"Number of CSV files extracted: {num_csv_files}")

from IPython.display import IFrame

IFrame(src=os.path.relpath(f"{csv_base_path}readme.html"), width=1000, height=600)

# Load sample CSV
df_temp = pd.read_csv('csv_files\On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2018_9.csv')

# Print the row and column length in the dataset, and print the column names.
df_shape = df_temp.shape
print(f'Rows and columns in one csv file is {df_shape}')

# Print the first 10 rows of the dataset.
print(df_temp.head(10))

# Print all the columns in the dataset. Use <dataframe>.columns to view the column names.
print('The column names are :')
print('#########')
for col in df_temp.columns:
    print(col)

# Print all the columns in the dataset that contain the word 'Del'. This will help you see how many columns have delay data in them.
del_columns = [col for col in df_temp.columns if 'Del' in col]

# Print the result
print("Columns containing 'Del':")
for col in del_columns:
    print(col)

# How many rows and columns does the dataset have?
# How many years are included in the dataset?
# What is the date range for the dataset?
# Which airlines are included in the dataset?
# Which origin and destination airports are covered?

print("The #rows and #columns are ", df_temp.shape[0] , " and ", df_temp.shape[1])
print("The years in this dataset are: ", df_temp['Year'].unique())
print("The months covered in this dataset are: ", df_temp['Month'].unique())
print("The date range for data is :" , min(df_temp['FlightDate']), " to ", max(df_temp['FlightDate']))
print("The airlines covered in this dataset are: ", list(df_temp['Reporting_Airline'].unique()))
print("The Origin airports covered are: ", list(df_temp['Origin'].unique()))
print("The Destination airports covered are: ", list(df_temp['Dest'].unique()))

# What is the count of all the origin and destination airports?
counts = pd.DataFrame({'Origin':df_temp['Origin'].value_counts(), 
                       'Destination':df_temp['Dest'].value_counts()})
counts

# Print the top 15 origin and destination airports based on number of flights in the dataset.
print("Top 15 origin and destination airports:")
print(counts.sort_values(by='Origin',ascending=False).head(15))

# Combine all CSV files
def combine_csv(csv_files, filter_cols, subset_cols, subset_vals, file_name):
    """
    Combine csv files into one Data Frame
    csv_files: list of csv file paths
    filter_cols: list of columns to filter
    subset_cols: list of columns to subset rows
    subset_vals: list of list of values to subset rows
    """
    # Create an empty dataframe
    df = pd.DataFrame() 
    
    # Loop through each csv file
    for file in csv_files:
        print(f'Reading {file}')
        temp_df = pd.read_csv(file)
        df_filter = temp_df[filter_cols]
        
        for col, vals in zip(subset_cols, subset_vals):
            df_filter = df_filter[df_filter[col].isin(vals)]
        
        df = pd.concat([df, df_filter], ignore_index=True)
    
    df.to_csv(file_name, index=False)
    print(f'Combined CSV saved as {file_name}')
    
    return df

# Combined CSV folder path
base_path = r'csv_files/'

csv_files = glob.glob(f"{base_path}/*.csv")


#cols is the list of columns to predict Arrival Delay 
cols = ['Year','Quarter','Month','DayofMonth','DayOfWeek','FlightDate',
        'Reporting_Airline','Origin','OriginState','Dest','DestState',
        'CRSDepTime','Cancelled','Diverted','Distance','DistanceGroup',
        'ArrDelay','ArrDelayMinutes','ArrDel15','AirTime']

subset_cols = ['Origin', 'Dest', 'Reporting_Airline']

# subset_vals is a list collection of the top origin and destination airports and top 5 airlines
subset_vals = [['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['UA', 'OO', 'WN', 'AA', 'DL']]

start = time.time()

combined_csv_filename = f"{base_path}combined_files.csv"

combined_df = combine_csv(csv_files, cols, subset_cols, subset_vals, combined_csv_filename)

print(f'csv\'s merged in {round((time.time() - start)/60,2)} minutes')

# Load the combined dataset
data = pd.read_csv('csv_files\combined_files.csv')

# Print the first 5 records.
print(data.head())

# How many rows and columns does the dataset have?
# How many years are included in the dataset?
# What is the date range for the dataset?
# Which airlines are included in the dataset?
# Which origin and destination airports are covered?

# to answer above questions, complete the following code
print("The #rows and #columns are ", data.shape[0] , " and ", data.shape[1])
print("The years in this dataset are: ", list(data['Year'].unique()))
print("The months covered in this dataset are: ", sorted(list(data['Month'].unique())))
print("The date range for data is :" , min(data['FlightDate']), " to ", max(data['FlightDate']))
print("The airlines covered in this dataset are: ", list(data['Reporting_Airline'].unique()))
print("The Origin airports covered are: ", list(data['Origin'].unique()))
print("The Destination airports covered are: ", list(data['Dest'].unique()))

# Define target column is_delay
data.rename(columns={'ArrDel15': 'is_delay'}, inplace=True)
print(data.columns)

# Count null values 
null_counts = data.isnull().sum(axis=0)
print(null_counts)

### Remove null columns for arrival delay details and airtime
null_re_dt = data[~data['ArrDelay'].isnull() & ~data['ArrDelayMinutes'].isnull()
                 & ~data['is_delay'].isnull() & ~data['AirTime'].isnull()]
                 
print(null_re_dt.shape)

# Get the hour of the day in 24-hour time format from CRSDepTime.
null_re_dt['DepHourofDay'] = null_re_dt['CRSDepTime']//100
print(null_re_dt[['CRSDepTime', 'DepHourofDay']].head())

# Check class delay vs. no delay
(null_re_dt.groupby('is_delay').size()/len(null_re_dt) ).plot(kind='bar')
plt.ylabel('Frequency')
plt.title('Distribution of classes')
plt.show()

# Which months have the most delays?
# What time of the day has the most delays?
# What day of the week has the most delays?
# Which airline has the most delays?
# Which origin and destination airports have the most delays?
# Is flight distance a factor in the delays?

viz_columns = ['Month', 'DepHourofDay', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest']
fig, axes = plt.subplots(3, 2, figsize=(20,20), squeeze=False)
# fig.autofmt_xdate(rotation=90)

for idx, column in enumerate(viz_columns):
    ax = axes[idx//2, idx%2]
    temp = null_re_dt.groupby(column)['is_delay'].value_counts(normalize=True).rename('percentage').\
    mul(100).reset_index().sort_values(column)
    sns.barplot(x=column, y="percentage", hue="is_delay", data=temp, ax=ax)
    plt.ylabel('% delay/no-delay')
    

plt.show()

sns.lmplot( x="is_delay", y="Distance", data=null_re_dt, fit_reg=False, hue='is_delay', legend=False)
plt.legend(loc='center')
plt.xlabel('is_delay')
plt.ylabel('Distance')
plt.show()

# Look at all the columns and what their specific types are.
print(null_re_dt.columns)
print(null_re_dt.dtypes)

# Filtering the required columns

data_orig = null_re_dt.copy()
new_data = null_re_dt[[ 'is_delay', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest','Distance','DepHourofDay']]
categorical_columns  = ['Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest', 'DepHourofDay']
for c in categorical_columns:
    new_data[c] = new_data[c].astype('category')
    
# use one-hot encoding for categorical columns
data_dummies = pd.get_dummies(new_data[categorical_columns], drop_first=True)
new_data = pd.concat([new_data, data_dummies], axis = 1)
new_data.drop(categorical_columns,axis=1, inplace=True)
print(new_data.shape)

# Check the length of the dataset and the new columnms.
print(new_data.columns)
print(new_data.dtypes)

# Rename the column is_delay to target
new_data.rename(columns = {'is_delay':'target'}, inplace=True )
print(new_data.columns)

# Save the combined new DataFrame to CSV file
new_data.to_csv('combined_csv_v1.csv', index=False)
print('File saved sucessfully as combined_csv_v1.csv')

### Model training and evaluation

# Split into features X and target y
X = new_data.drop('target', axis=1)
y = new_data['target']

# Train test split
train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, 
                                                                  random_state=42, stratify=y)

# Baseline classification model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(train_data, train_label)

train_pred = lr_model.predict(train_data)

cm_train = confusion_matrix(train_label, train_pred)
print(cm_train)

train_accuracy = accuracy_score(train_label, train_pred)
train_precision = precision_score(train_label, train_pred)
train_recall = recall_score(train_label, train_pred)
train_auc = roc_auc_score(train_label, train_pred)
print(f"Accuracy: {train_accuracy:.3f}")
print(f"Precision: {train_precision:.3f}")
print(f"Recall: {train_recall:.3f}")
print(f"AUC: {train_auc:.3f}")


target_predicted = lr_model.predict(test_data)

# Model evaluation
# Confusion Matrix plot
def plot_confusion_matrix(test_labels, target_predicted):
    cm = confusion_matrix(test_labels, target_predicted)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()

    # Compute metrics
    accuracy = accuracy_score(test_labels, target_predicted)
    precision = precision_score(test_labels, target_predicted)
    recall = recall_score(test_labels, target_predicted)   
    specificity = tn / (tn + fp)

    # Print metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall : {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")

# --- ROC Curve Function ---
def plot_roc(test_labels, target_predicted):
    fpr, tpr, thresholds = roc_curve(test_labels, target_predicted)
    auc = roc_auc_score(test_labels, target_predicted)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.show()



# Plot Confusion Matrix
plot_confusion_matrix(test_label, target_predicted)

# Plot ROC Curve
plot_roc(test_label, target_predicted)
