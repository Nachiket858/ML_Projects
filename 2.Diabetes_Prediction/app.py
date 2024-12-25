from flask import Flask, render_template, request, redirect, url_for
import pickle
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model and scaler
with open(r'E:\Python\Myenv\Data_Science\ML_Projects\2.Diabetes_Prediction\Models\svc_classifire.pkl', 'rb') as file:
    svc_classifire = pickle.load(file)

# with open(r'E:\Python\Myenv\Data_Science\ML_Projects\2.Diabetes_Prediction\Models\Scalar.pkl', 'rb') as file:
#     scalar = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data from the user
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])

        # Prepare the data for prediction
        data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
        # newdata = scalar.transform(data)  # Scale the data using the scalar

        # Get the prediction
        result = svc_classifire.predict(data)

        # Render the result on the prediction page
        if result == 1:
            prediction = 'You are likely to have diabetes.'
        else:
            prediction = 'You are unlikely to have diabetes.'

        return render_template('predict.html', prediction=prediction)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
