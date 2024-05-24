import numpy as np
from sklearn.preprocessing import StandardScaler

from flask import Flask, render_template, request
import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
from nltk.stem.porter import PorterStemmer

scaler = StandardScaler()

# ps = PorterStemmer()

# importing model
model = pickle.load(open('model_dib.pkl', 'rb'))


# function to transfrom user input



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('dib1.html')


@app.route('/predict', methods=['POST'])
def predict():
    ui1 = request.form['a']
    ui2 = request.form['b']
    ui3 = request.form['c']
    ui4 = request.form['d']
    ui5 = request.form['e']
    ui6 = request.form['f']
    ui7 = request.form['g']
    ui8 = request.form['h']

    ui_list = (ui1, ui2, ui3, ui4, ui5, ui6, ui7, ui8)

    # changing the input_data into numpy array
    inp_data = np.asarray(ui_list)

    # reshape the array as we are predicting for one instance
    inp_reshaped = inp_data.reshape(1, -1)

    # standardize the input data
    std_data = scaler.fit_transform(inp_reshaped)
    print(std_data)

    prediction = model.predict(std_data)
    print(prediction)

    res = prediction[0]

    return render_template('output.html', data=res)


if __name__ == "__main__":
    app.run(debug=True)
