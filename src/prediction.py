import pandas as pd
import pickle

def load_model():
    with open('../model/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(new_data):
    model = load_model()
    predictions = model.predict(new_data)
    return predictions


