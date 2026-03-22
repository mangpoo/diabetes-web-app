import numpy as np
import joblib
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    preg    = StringField('# Pregnancies',  validators=[DataRequired()])
    glucose = StringField('Glucose',        validators=[DataRequired()])
    blood   = StringField('Blood pressure', validators=[DataRequired()])
    skin    = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin',        validators=[DataRequired()])
    bmi     = StringField('BMI',            validators=[DataRequired()])
    dpf     = StringField('DPF Score',      validators=[DataRequired()])
    age     = StringField('Age',            validators=[DataRequired()])
    submit  = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])

        scaler = joblib.load('pima_scaler.pkl')
        model  = joblib.load('pima_model.pkl')

        X_scaled = scaler.transform(X_test)
        prob = model.predict_proba(X_scaled)[0][1]
        res  = float(round(prob * 100))

        return render_template('result.html', res=res)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()