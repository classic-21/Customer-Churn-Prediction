from flask import Flask, request, render_template, send_file
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler using joblib
model = joblib.load('model.pkl')
scaler = joblib.load('scalar.pkl')

# Selected features for the model
selected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges', 'HasInternet', 'TotalActiveServices'
]

# Route for the homepage with a form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data and convert to a pandas DataFrame
        data = request.form
        
        # Prepare input data based on the selected features
        input_data = {
            'gender': [int(data.get('gender'))],
            'SeniorCitizen': [int(data.get('SeniorCitizen'))],
            'Partner': [int(data.get('Partner'))],
            'Dependents': [int(data.get('Dependents'))],
            'tenure': [float(data.get('tenure'))],
            'PhoneService': [int(data.get('PhoneService'))],
            'MultipleLines': [int(data.get('MultipleLines'))],
            'InternetService': [int(data.get('InternetService'))],
            'OnlineSecurity': [int(data.get('OnlineSecurity'))],
            'OnlineBackup': [int(data.get('OnlineBackup'))],
            'DeviceProtection': [int(data.get('DeviceProtection'))],
            'TechSupport': [int(data.get('TechSupport'))],
            'StreamingTV': [int(data.get('StreamingTV'))],
            'StreamingMovies': [int(data.get('StreamingMovies'))],
            'Contract': [int(data.get('Contract'))],
            'PaperlessBilling': [int(data.get('PaperlessBilling'))],
            'PaymentMethod': [int(data.get('PaymentMethod'))],
            'MonthlyCharges': [float(data.get('MonthlyCharges'))],
            'TotalCharges': [float(data.get('TotalCharges'))],
            'HasInternet': [int(data.get('HasInternet'))],
            'TotalActiveServices': [int(data.get('TotalActiveServices'))]
        }
        
        # Create a DataFrame from input data
        df = pd.DataFrame(input_data)

        df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        
        # Model prediction
        prediction = model.predict(df)
        
        # Return result
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
