import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data():
    # load dataset from sklearn
    data = load_breast_cancer()
    x = data.data
    y = data.target
    
    # transform labels to -1 and 1 for svm
    y = np.where(y == 0, -1, 1)
    
    # scale data for better convergence
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )
    
    print("data loaded successfully")
    print("train shape:", x_train.shape)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    prepare_data()