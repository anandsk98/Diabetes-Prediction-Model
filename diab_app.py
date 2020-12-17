from flask import Flask, render_template, request
import pickle
import numpy as np

filename='diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def man():
   if request.method == 'POST':
       pregnent =int(request.form['pregnencies'])
       Glucose =int(request.form['glucose'])
       Bloodpressure =int(request.form['bloodpressure'])
       St =int(request.form['skinthickness'])
       Insulin =int(request.form['insulin'])
       BMI=float(request.form['bmi'])
       DPF=float(request.form['dpf'])
       Age =float(request.form['age'])

       arr = np.array([[pregnent,Glucose,Bloodpressure,St,Insulin,BMI,DPF,Age]])
       pred = classifier.predict(arr)
        
       return render_template('after.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)

