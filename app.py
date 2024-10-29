from flask import Flask, request, render_template
import joblib
import pandas as pd

mymodel = joblib.load('mymodel.joblib')
app = Flask(__name__) #__name__ = app
model_class = {
    '0':'setosa',
    '1':'versicolor',
    '2':'virginica'
}
@app.route("/", methods=['GET','POST']) #decorador para ruta raiz
def index():
    if request.method == 'POST':
        val_1 = float(request.form['val1'])
        val_2 = float(request.form['val2'])
        val_3 = float(request.form['val3'])
        val_4 = float(request.form['val4'])
        my_df = pd.DataFrame([[val_1, val_2, val_3, val_4]], columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'])
        prediction = str(mymodel.predict(my_df)[0])
        pred_class = model_class[prediction]
    else:
        pred_class = None
    return render_template('index.html', prediction = pred_class)

@app.route('/child')
def hello_child():
    return 'Hello child'



