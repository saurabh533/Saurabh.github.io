from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

app = Flask(__name__)


scaler = joblib.load('scaler.pkl')
model = pickle.load(open('random_forest_pickle', 'rb'))

@app.route('/')
def index():
    return render_template("index.html", churn_prediction=None)


def predict_churn(input_data):
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform([[
        input_data['age'],
        input_data['subscription_length'],
        input_data['monthly_bill'],
        input_data['total_usage_gb']
    ]])


    gender_encoded = [1, 0]  # Initialize as [0, 0] for Female
    if input_data['gender'] == 'Male':
        gender_encoded = [0, 1]  # Set to [0, 1] for Male

    # Encode location as one-hot encoding
    locations = [
        'Chicago',
        'Houston',
        'Miami',
        'Los Angeles',
        'New York'

    ]
    location_encoded = [0] * len(locations)
    location_encoded[locations.index(input_data['location'])] = 1

    # Create a DataFrame from the processed input data
    input_df = pd.DataFrame.from_dict({
        'Age': scaled_features[0][0],
        'Subscription_Length_Months': scaled_features[0][1],
        'Monthly_Bill': scaled_features[0][2],
        'Total_Usage_GB': scaled_features[0][3],
        'Gender_Female': gender_encoded[0],
        'Gender_Male': gender_encoded[1],
        'Location_Chicago':location_encoded[0],
        'Location_Houston':location_encoded[1],
        'Location_Los Angeles':location_encoded[2],
        'Location_Miami':location_encoded[3],
        'Location_New York':location_encoded[4]

    }, orient='index').T


    churn_prediction = model.predict(input_df)
    if churn_prediction[0] == 0:
        churn_prediction = 'No_churn'
    else:
        churn_prediction = 'Yes'
    return churn_prediction

@app.route('/predict_churn', methods=['POST'])
def predict_churn_():# Extract user input data from the form
    age = float(request.form['age'])
    subscription_length = float(request.form['subscription_length'])
    monthly_bill = float(request.form['monthly_bill'])
    total_usage_gb = float(request.form['total_usage_gb'])
    gender = request.form['gender']
    location = request.form['location']

        # Create a dictionary with the user input
    input_data = {
            'age': age,
            'subscription_length': subscription_length,
            'monthly_bill': monthly_bill,
            'total_usage_gb': total_usage_gb,
            'gender': gender,
            'location': location
    }


    churn_prediction_model = predict_churn(input_data)

    return render_template('index.html', churn_prediction=churn_prediction_model)



if __name__ == '__main__':
    app.run(debug=True)
