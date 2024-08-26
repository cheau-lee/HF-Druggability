import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

### PROCESSED FEATURES ###

# Load the data
file_path = '/Users/cheaulee/Desktop/hfproject/DATA/processed_feature_table.tsv'
df = pd.read_csv(file_path, sep='\t')

# Print the number of features
num_features = df.shape[1]
print(f'The number of features in the dataset: {num_features}')

# File paths
file_path = '/Users/cheaulee/Desktop/hfproject/DATA/processed_feature_table.tsv'
output_file = '/Users/cheaulee/Desktop/hfproject/PF/featureslist.txt'

# Read the column titles from the TSV file
with open(file_path, 'r') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    column_titles = next(reader)

# Write the column titles to a text file
with open(output_file, 'w') as txt_file:
    for title in column_titles:
        txt_file.write(title + '\n')

print(f"Column titles have been saved to {output_file}")

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Identify binary columns
binary_cols = [col for col in numeric_cols if np.array_equal(numeric_cols[col].dropna().unique(), [0, 1])]

# Count the number of binary columns
binary_col_count = len(binary_cols)

print(f'Number of binary columns: {binary_col_count}')

# Randomly select 38 features (columns) from the dataframe
random_features = random.sample(list(df.columns), 38)

# Visualise the distribution of each selected feature
plt.figure(figsize=(25, 30))
for i, feature in enumerate(random_features, 1):
    plt.subplot(9, 5, i)  
    df[feature].hist(bins=20)
    plt.title(feature)

plt.tight_layout()
plt.show()

# Drop binary features (0 and 1)
numeric_cols = df.select_dtypes(include=[np.number])
non_binary_cols = [col for col in numeric_cols if not np.array_equal(numeric_cols[col].dropna().unique(), [0, 1])]

# Calculate skewness for each non-binary numeric column and count how many have positive skew
positive_skew_count = sum(df[col].skew() > 0 for col in non_binary_cols)

print(f'Number of columns with positive skew: {positive_skew_count}')




### DRUGGABILITY FEATURES ###


# Load the data
file_path = '/Users/cheaulee/Desktop/hfproject/DATA/druggability_feature_table.tsv'
df = pd.read_csv(file_path, sep='\t')

# Print the number of features
num_features = df.shape[1]
print(f'The number of features in the dataset: {num_features}')


# File paths
file_path = '/Users/cheaulee/Desktop/hfproject/DATA/druggability_feature_table.tsv'
output_file = '/Users/cheaulee/Desktop/hfproject/DF/featureslist.txt'

# Read the column titles from the TSV file
with open(file_path, 'r') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    column_titles = next(reader)

# Write the column titles to a text file
with open(output_file, 'w') as txt_file:
    for title in column_titles:
        txt_file.write(title + '\n')

print(f"Column titles have been saved to {output_file}")

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Identify binary columns
binary_cols = [col for col in numeric_cols if np.array_equal(numeric_cols[col].dropna().unique(), [0, 1])]

# Count the number of binary columns
binary_col_count = len(binary_cols)

print(f'Number of binary columns: {binary_col_count}')

# Randomly select 43 features (columns) from the dataframe
random_features = random.sample(list(df.columns), 43)

# Visualise the distribution of each selected feature
plt.figure(figsize=(25, 30))
for i, feature in enumerate(random_features, 1):
    plt.subplot(9, 5, i)  
    df[feature].hist(bins=20)
    plt.title(feature)

plt.tight_layout()
plt.show()

# Drop binary features (0 and 1)
numeric_cols = df.select_dtypes(include=[np.number])
non_binary_cols = [col for col in numeric_cols if not np.array_equal(numeric_cols[col].dropna().unique(), [0, 1])]

# Calculate skewness for each non-binary numeric column and count how many have positive skew
positive_skew_count = sum(df[col].skew() > 0 for col in non_binary_cols)

print(f'Number of columns with positive skew: {positive_skew_count}')
