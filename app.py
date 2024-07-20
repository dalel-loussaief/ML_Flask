from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import pandas as pd
import json


from model import cleaned_df,model_rf

app = Flask(__name__)

# Load the trained RandomForestRegressor model
# model_path = 'random_forest_model.pkl'
# model_rf = joblib.load(model_path)



# Define the endpoint for predicting house prices
@app.route('/predict_price', methods=['POST'])
def predict_price():
    # Get the request data (house features)
    house_data = request.json


    
    # Create a DataFrame from the received data
    house_features = pd.DataFrame([house_data])


    
    # # Ensure all other categorical columns are set to 0
    for column in cleaned_df.columns:
        if column not in house_features.columns:
            house_features[column] = 0

    # # Reorder columns to match the model's input format
    house_features = house_features[cleaned_df.drop('price', axis=1).columns]


    
    # # Make predictions using the trained Random Forest model
    predicted_price = model_rf.predict(house_features)

    print(predicted_price)

    
    # # Round each element of the predicted price array to two decimal places
    predicted_price_rounded = np.round(predicted_price, 2)
    
    # Return the predicted price as JSON response
    return jsonify({"predicted_price": predicted_price_rounded.tolist()})

if __name__ == '__main__':
    app.run(debug=True)