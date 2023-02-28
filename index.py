from flask import Flask, flash, redirect, render_template, jsonify, request, url_for
from werkzeug.utils import secure_filename

import numpy as np
import os
import glob
import matplotlib.pyplot as mp
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as om
import torchvision as tv
import torch.utils.data as dat
import markdown

from classes.MedNet import MedNet

toTensor = tv.transforms.ToTensor()
def scaleImage(x):          # Pass a PIL image, return a tensor
    y = toTensor(x)
    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        # Subtract the mean value of the image
    return z

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

dataDir = 'static/uploads'               
label_names = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
model = torch.load("static/saved_model")

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def accueil():  
    
    files = glob.glob('static/uploads/*')
    for f in files:
        os.remove(f)
    return render_template('index.html')

@app.route("/upload")
def upload():  
    
    files = glob.glob('static/uploads/*')
    for f in files:
        os.remove(f)
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('image')
    for file in files:
        if file:
            # Enregistrer le fichier sur le serveur
            file.save('static/uploads/' + file.filename)

    predict_dict = []
    imageFilesList = os.listdir(dataDir)

    for image in imageFilesList:
        file = []
        imageTensor = torch.stack([scaleImage(Image.open('static/uploads/' + image))])
        output = model(imageTensor)
        predicted_class = torch.max(output, dim=1)[1]
        label = label_names[predicted_class]
        file.append(image)
        file.append(predicted_class.item())
        file.append(label)
        predict_dict.append(file)
        
    return render_template('predict.html', filename=predict_dict)

@app.route("/notebook")
def notebook():    
    with open('static/readme/README.md', 'r') as f:
        readme_content = f.read()

    html_content = markdown.markdown(readme_content)
    return render_template('notebook.html', content=html_content)

@app.route("/explication")
def explication():    
    return render_template('explication.html')

if __name__ == '__main__':
    app.run(debug=True) 


