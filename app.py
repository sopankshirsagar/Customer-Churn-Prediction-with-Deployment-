import pickle
from flask import Flask,request,app,jsonify ,render_template, url_for 
import pandas as pd
import numpy as np 


app=Flask(__name__)
model=pickle.load(open('customer churn.pkl','rb'))

@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(f"\n{'-~'*50}\n",data,f"\n{'-~'*50}\n")

    new_data= [list(data.values())]
    output=model.predict(new_data)[0]
   
   # conver into int   
    if isinstance(output, np.int64) :
        output=int(output)
    
    return jsonify(output)
 

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_feature=[np.array(data)]
    print(data)
    
    output = model.predict(final_feature)[0]
    print(output)
    if output ==1 :
        output="Exited (1)"
    else :
        output="Not Exited (0)"
    return render_template('Home.html',prediction_text="customer is :>  {}".format(output))

 

 
if __name__=='__main__':
    app.run(debug=True)
      