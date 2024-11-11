import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

def train_model():
    df = pd.read_csv('../data/processed/drought_data.csv')
    independent_variables = df.drop(['score', 'fips', 'date'], axis=1)
    target = df['score']

    X_train, X_test, y_train, y_test = train_test_split(independent_variables, target, test_size=0.2, random_state=0)
    
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    
    with open('../model/random_forest.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
