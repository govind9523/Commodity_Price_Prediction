from flask import Flask, render_template, request
import pandas as pd
import pickle


app = Flask(__name__)

# Load your dataset into a DataFrame (example path)
df_melted = pd.read_csv('df_melted.csv')

# Create the commodity mapping
commodity_mapping = {
    'Wheat': 1101010102,
    'Ragi': 1101010107,
    'a2. PULSES': 1101010200,
    'Gram': 1101010201,
    'Peas/Chawali': 1101010206,
    'b1. VEGETABLES': 1101020100,
    'Potato': 1101020101,
    'Sweet Potato': 1101020102,
    'Ginger (Fresh)': 1101020105
}

# Load the trained model from the pickle file
with open('trained_model (2).pkl', 'rb') as file:
    best_model = pickle.load(file)

# Make sure you fit on the original Price column from your dataset


# Prepare features from your dataset
df_melted.columns = ['COMM_NAME', 'COMM_CODE', 'COMM_WT', 'Quarter', 'Price', 'Quarter_numeric']
features = df_melted[['COMM_CODE', 'COMM_WT', 'Quarter_numeric']]
target = df_melted['Price']
features = pd.get_dummies(features, drop_first=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    commodity = None
    year = None
    quarter = None

    if request.method == 'POST':
        comm_name = request.form['commodity']
        weight = float(request.form['wt'])
        year = int(request.form['year'])
        quarter = int(request.form['quarter'])

        # Calculate quarter_numeric as a float
        quarter_numeric = year + (quarter / 10.0)

        comm_code = commodity_mapping.get(comm_name)
        if comm_code is not None:
            predicted_price = predict_price(comm_code, weight, quarter_numeric)
           
            return render_template('1.html', predicted_price=predicted_price, commodity=comm_name, year=year,
                                   quarter=quarter)

    return render_template('1.html')


def predict_price(comm_code, weight, quarter_numeric):
    input_data = pd.DataFrame({
        'COMM_CODE': [comm_code],
        'COMM_WT': [weight],
        'Quarter_numeric': [quarter_numeric]
    })

    # Preprocess the input data (one-hot encoding)
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Align input data with training data
    input_data = input_data.reindex(columns=features.columns, fill_value=0)

    # Make prediction
    predicted_price = best_model.predict(input_data)

    return predicted_price[0]


app.run(debug=True)
