import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
import joblib
from sklearn.preprocessing import MinMaxScaler
# Image transformation for preprocessing
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, url_for , send_file , Response
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import io
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from skimage import color, filters, measure
from skimage import io as skio, color, filters, measure, morphology, segmentation
import matplotlib
import pickle
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Initialize Flask app
app = Flask(__name__)



upload_folder = os.path.join(f"uploads")
os.makedirs(upload_folder, exist_ok=True)



opg = os.path.join(f"uploads/opg")
os.makedirs(opg, exist_ok=True)



bitewing = os.path.join(f"uploads/bitewing")
os.makedirs(bitewing, exist_ok=True)



app.config.update(SECRET_KEY="rms-oral")





login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"{self.id}"


# Sample hardcoded user data (username and plain text password)
users = {
    "rambod": "saboori_123123"  # Plain text password
}


# Load user from the ID
@login_manager.user_loader
def load_user(userid):
    return User(userid)


@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.id)


@app.route("/login")
def login():
    return render_template("sign-in.html")


@app.route("/login", methods=["POST"])
def loggin():
    username = request.form["username"]
    password = request.form["password"]

    # Check if the username exists and password matches in plain text
    if username in users and users[username] == password:
        user = User(username)
        login_user(user)  # Log the user in
        return redirect(url_for("dashboard"))  # Redirect to the protected dashboard
    else:
        return render_template("sign-in.html", error="Username or password is invalid")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# Error handler for unauthorized access
@app.errorhandler(401)
def page_not_found(e):
    return Response("""
                    <html><center style='background-color:white;'>
                    <h2 style='color:red;'>Login failed</h2>
                    <h1>Error 401</h1>
                    </center></html>""")



""" This Route Get histopathologic_cancer Detection. Give Image Like Samples
to return Valid Response"""


@app.route("/eye_disease")
def histopathologic_oral_cancer_get():
    return render_template("eye_disease.html")


# Define the CNN model for cancer detection
class SimpleCNN_eye_disease(nn.Module):
    def __init__(self):
        super(SimpleCNN_eye_disease, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 4)  # Second fully connected layer for the output (4 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output of the conv layer
        x = F.relu(self.fc1(x))  # Pass through the first fully connected layer
        x = self.fc2(x)  # Pass through the second fully connected layer
        return x

    


# Initialize model and load weights
# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eye_disease_model = SimpleCNN_eye_disease().to(device)
eye_disease_model.load_state_dict(torch.load('eye_classifier.pth', map_location=device))
eye_disease_model.eval()

# Define the transformation for the input image
eye_disease_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to expected input size of the model
    transforms.ToTensor(),  # Convert the image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Define the route for prediction
@app.route('/eye_disease', methods=['POST'])
def eye_disease_predict():
    file = request.files['eye_image']
    
    if file.filename == '':
        return {'error': 'No selected file'}
    
    try:
        # Open and transform the image
        image = Image.open(file.stream)
        image = eye_disease_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Model prediction
        with torch.no_grad():
            outputs = eye_disease_model(image)
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        
        # Class names for prediction
        class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        predicted_class = class_names[predicted.item()]
        
        # Return the prediction result as part of the rendered HTML
        return render_template("eye_disease.html", prediction=predicted_class)

    except Exception as e:
        return {'error': str(e)}







@app.route("/about_us")
def about_us():
    return render_template("profile.html")


@app.route("/contact_us")
def contact_us():
    return render_template("contact_us.html")



@app.route("/tables")
def tables_view():
    return render_template("tables.html")


if __name__ == '__main__':
    app.run(debug=True , port = 5005)
