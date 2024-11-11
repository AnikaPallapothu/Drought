
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Drought Score Predictor", layout="wide")

# Header with title and subtitle
st.title("Drought Score Prediction System")
st.markdown("Upload a testing file, and let the model predict the drought score along with insightful visualizations.")

# File uploader for test data
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file:
    # Load the model
    @st.cache(allow_output_mutation=True)
    def load_model():
        with open('./model/random_forest.pkl', 'rb') as f:
            return pickle.load(f)
    
    model = load_model()

    # Read the uploaded CSV file
    test_data = pd.read_csv(uploaded_file)
    
    # Display a preview of the uploaded data
    st.write("Preview of the uploaded data:")
    st.dataframe(test_data.head())

    # Automatically identify target and feature columns based on common names or inference
    target_col = None
    potential_target_columns = ['score', 'label', 'target']  # Potential target column names
    for col in potential_target_columns:
        if col in test_data.columns:
            target_col = col
            break
    
    # Separate features and target if a target column is found
    if target_col:
        y_true = test_data[target_col]
        X_test = test_data.drop(columns=[target_col, 'fips', 'date'], errors='ignore')  # Drop non-feature columns
    else:
        st.error("Target column (e.g., 'score', 'label', or 'target') not found in the dataset.")
        y_true = None
        X_test = test_data.drop(columns=['fips', 'date'], errors='ignore')  # Assume entire dataset is for prediction

    # Ensure that the test data columns match those used during training
    st.write("### Features used for prediction:")
    st.dataframe(X_test.head())  # Display the features being used for prediction

    # Predicting using the model
    y_pred = model.predict(X_test)

    # Display the predictions in a new DataFrame
    st.write("### Predictions")
    result_df = test_data.copy()
    result_df['Predicted_Score'] = y_pred
    st.dataframe(result_df)

    # Save the predictions to a CSV file
    result_df.to_csv('predictions.csv', index=False)
    st.markdown(f"[Download Predictions CSV](predictions.csv)")

    if y_true is not None:
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        st.markdown("### Evaluation Metrics")
        st.write(f"Accuracy: **{accuracy * 100:.2f}%**")
        st.write(f"Precision: **{precision * 100:.2f}%**")
        st.write(f"Recall: **{recall * 100:.2f}%**")

        # Confusion Matrix Visualization
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    # Predicted Score Distribution
    st.markdown("### Score Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(y_pred, bins=10, kde=True, ax=ax)
    ax.set_title("Predicted Score Distribution")
    st.pyplot(fig)

    # Bar plot of predictions
    st.markdown("### Bar Chart of Scores")
    fig, ax = plt.subplots(figsize=(7, 4))
    pd.Series(y_pred).value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Bar Chart of Predicted Scores")
    st.pyplot(fig)

    # Pie chart of predicted scores
    st.markdown("### Pie Chart of Predicted Scores")
    fig, ax = plt.subplots(figsize=(7, 4))
    pd.Series(y_pred).value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
    ax.set_ylabel("")
    st.pyplot(fig)
else:
    st.markdown("Please upload a CSV file to start the prediction process.")

