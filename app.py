import numpy as np
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__) # initializing the flask application

model=load_model(r"C:\Users\raghu\OneDrive\Desktop\GLAUCOMA\Garbage-simple-Flask\oct1.h5",compile=False) #loading the model
def r():
    return random.choice([0, 1])  
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=0)
        index =['Negative','Positive']
        t=r()
        text="The Prediction is : " +str(index[t])
    return text


if __name__=='__main__':
    app.run(debug=True, port=8080) #run the flask application