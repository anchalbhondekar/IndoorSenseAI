import pandas as pd

def load_data():
    train = pd.read_csv("C:/Users/ancha/Desktop/projects/IndoorSense AI/data/trainingData.csv")
    test = pd.read_csv("C:/Users/ancha/Desktop/projects/IndoorSense AI/data/validationData.csv")
    
    return train, test