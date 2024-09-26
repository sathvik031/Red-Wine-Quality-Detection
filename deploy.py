from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('savedmodel.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        # Retrieve input from the form
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        ph = float(request.form['ph'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        # Make the prediction using the model
        result = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                 free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])[0]

        # Render the result on the index.html page
        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
