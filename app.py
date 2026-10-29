from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
# feature colunm
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get values from the HTML form
    school = request.form['school']
    age = int(request.form['age'])
    sex = request.form['sex']
    address = request.form['address']
    Mjob = request.form['Mjob']
    Fjob = request.form['Fjob']
    reason = request.form['reason']
    guardian = request.form['guardian']
    traveltime = int(request.form['traveltime'])
    studytime = int(request.form['studytime'])
    failures = int(request.form['failures'])
    schoolsup = request.form['schoolsup']
    famsup = request.form['famsup']
    paid = request.form['paid']
    activities = request.form['activities']
    internet = request.form['internet']
    freetime = int(request.form['freetime'])
    absences = int(request.form['absences'])

    # Create input using all features used during training
    student = pd.DataFrame([{
        'school': school,
        'sex': sex,
        'age': age,
        'address': address,
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': 2,
        'Fedu': 2,
        'Mjob': Mjob,
        'Fjob': Fjob,
        'reason': reason,
        'guardian': guardian,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'nursery': 'yes',
        'higher': 'yes',
        'internet': internet,
        'romantic': 'no',
        'famrel': 4,
        'freetime': freetime,
        'absences': absences
    }])

    # Convert categorical columns into dummy variables
    categorical_columns = [
        'school', 'sex', 'address', 'famsize', 'Pstatus',
        'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
        'famsup', 'paid', 'activities', 'nursery', 'higher',
        'internet', 'romantic'
    ]

    student = pd.get_dummies(
        student,
        columns=categorical_columns,
        dtype=int
    )

    # Match the columns used while training the model
    student = student.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Make prediction
    prediction = model.predict(student)[0]

    if str(prediction).lower() in ['1', '1.0', 'true', 'yes', 'pass', 'passed']:
        prediction_text = 'Student is predicted to PASS'
    else:
        prediction_text = 'Student is predicted to FAIL'

    # Display result
    return render_template('result.html',
        school=school,
        age=age,
        studytime=studytime,
        failures=failures,
        absences=absences,
        prediction_text=prediction_text
    )


if __name__ == '__main__':
    app.run(debug=True)
