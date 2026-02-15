import sys
import os
sys.path.append("..//")
from sklearn.linear_model import LogisticRegression
import joblib
from utils.data_import import get_mobile_train_data
from sklearn.model_selection import train_test_split

model_dir = "../pkl"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "logisticModel.pkl")

def data_preprocessing(feature_data, target_data):
    # Split features and target
    X = feature_data  # replace 'target' with your column name
    y = target_data

    return X,y

def model_train(X_train, y_train):
    # Create model
    model = LogisticRegression(
        C=0.1,
        max_iter=2000
    )
    # Fit model
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def main():
    X_train, y_train = get_mobile_train_data()
    #X_train, y_train = data_preprocessing(X_train, y_train)
    #train model and predict
    model = model_train(X_train, y_train)
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()