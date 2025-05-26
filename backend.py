from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler

# Max Value:  19731.4022877729 Min Value:  0.0
# (şuanki değer - min / max - min ) * (1 - 0) + 0 = normalize

app = Flask(__name__)
CORS(app)
model_path = 'D:\\Jupyter Projects\\deployment\\best_model.joblib'
scaler_path = 'D:\\Jupyter Projects\\deployment\\scaler.joblib'
feature_names_path = 'D:\\Jupyter Projects\\deployment\\feature_names.joblib'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(feature_names_path)

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # print(df.T)
    print(feature_names)
    

    df = df[feature_names]  # Ensure exact order

    print(df.T)
    print(scaler)

    df_scaled_array = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled_array, columns=feature_names)

    norm_prediction = model.predict(df_scaled)
    prediction = norm_prediction[0] * 1

    return jsonify({'prediction': prediction})

@app.route('/', methods=['GET'])
def home():
     
     print(feature_names)
     return jsonify({
        "message": "Forest Fire Prediction API",
         "usage": "Send a POST request to /predict with member features as JSON"
})


if __name__ == '__main__':
    app.run(debug=True)