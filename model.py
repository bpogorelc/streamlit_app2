import streamlit as st
import pandas as pd
import plotly.express as px    
from sklearn.ensemble import RandomForestRegressor
import pickle
# Function to train and save the model
def train_and_save_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Load the CSV file
    file_path = 'global_development_data.csv'
    data = pd.read_csv(file_path)

    # Prepare the data for training
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    X = data[features]
    y = data['Life Expectancy (IHME)']

       # Train the RandomForest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the model to a file
    with open('life_expectancy_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model trained and saved successfully.")
