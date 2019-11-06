# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:29:36 2019

@author: ayrem
"""
import numpy as np
from flask import Flask,request, jsonify, render_template
import pickle
from churn import df

app = Flask(__name__)

a = pickle.load(open("ram_model.pkl","rb"))
"""
@app.route('/')
def home():
    return render_template("home_template.html")
"""

@app.route('/predict', methods=["POST"])
def predict():
    
    global a
    data = request.form 
    if data["data_2"]=="France":
        data["data_2"]="0"
    elif data["data_2"]=="Germany":
        data["data_2"]="0.5"
    elif data["data_2"]=="Spain":  
        data["data_2"]="1"
        
    if data["data_3"]=="Male":
        data["data_3"]="1"
    elif data["data_3"]=="Female":
        data["data_3"]="0"
        
    if data["data_8"]=="No":
        data["data_8"]="0"
    elif data["data_8"]=="Yes":
        data["data_8"]="1"
        
    if data["data_9"]=="No":
        data["data_9"]="0"
    elif data["data_9"]=="Yes":
        data["data_9"]="1"
    """
    d = data["data_1"]
    d = (float(d)-350)/500
    data["data_1"] = str(d)
    """
    range(len(data))   
    data["data_1"] = ((data["data_1"])-350)/500
    """
    data["data_4"] = (data["data_4"]-18)/74
    data["data_5"] = (data["data_5"]-0)/10
    data["data_6"] = (data["data_6"]-0)/250898.090
    data["data_7"] = (data["data_7"]-1)/3
    data["data_10"] = (data["data_10"]-11.580)/199980.9
    """
        
    data = [data["data_1"], data["data_2"], data["data_3"], data["data_4"], data["data_5"], data["data_6"], data["data_7"], data["data_8"], data["data_9"], data["data_10"] ]    
    res = list(a.predict([data])) 
    return jsonify([res]) 

@app.route('/', methods=["GET"])
def home():
    # arayuzu koddan ayiralim
    return render_template("home_template.html")
app.run()