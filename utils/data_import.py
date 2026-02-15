import pandas as pd

# Set the path to the file you'd like to load
train_file_path = "../train.csv"
test_file_path="../test.csv"
targetColumn = "price_range"

def get_feature_target_data(df):    
    X = df.drop(targetColumn, axis=1)   # replace 'target' with your column name
    y = df[targetColumn]
    return  X, y

def read_csv(file_path):
    return pd.read_csv(file_path)

def get_mobile_train_data():
    return get_feature_target_data(read_csv(train_file_path))

def get_mobile_test_data():
    return read_csv(test_file_path)
