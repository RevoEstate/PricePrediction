from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import os
from model import RevoNeuralNetwork
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Property(BaseModel):
    Bedrooms: int
    Bathrooms: int
    Land_Area: float
    Year: int
    Status: Literal['For Rent', 'For Sale']
    Furnished: Literal['Yes', 'No']
    Address: Literal['Addis Ketema', 'Akaky Kaliti', 'Arada', 'Bole', 'Gullele', 'Kirkos', 
                     'Kolfe Keranio', 'Lideta', 'Nifas Silk-Lafto', 'Yeka', 'Lemi Kura']
    Property_Type: Literal['Apartment', 'Villa']

class Encode:
    def __init__(self, filename: str, value: str, column_name: str):
        self.filename = os.path.join(BASE_DIR, filename)
        self.value = value
        self.column_name = column_name  
        self.encoded_data = None

    def encode(self):
        self.encoder = joblib.load(self.filename)
        input_data = pd.DataFrame([[self.value]], columns=[self.column_name])
        self.encoded_data = self.encoder.transform(input_data)

    def get_encoded_data(self):
        return self.encoded_data
def predictPrice(encoded_data):
        # Load the model /home/jibril/Documents/REvoEstatePricePrediction/Trainig/state_dictmodel.pth
        mean=torch.load(os.path.join(BASE_DIR, "Trainig/mean.pt"))
        std=torch.load(os.path.join(BASE_DIR, "Trainig/std.pt"))
        Y_mean=torch.load(os.path.join(BASE_DIR, "Trainig/Y_mean.pt"))
        Y_std=torch.load(os.path.join(BASE_DIR, "Trainig/Y_std.pt"))
        input_tensor = torch.tensor(encoded_data, dtype=torch.float32).unsqueeze(0)
        input_tensor_scaled = (input_tensor - mean) / std
        # Load the model

        loadedmodel = RevoNeuralNetwork()
        loadedmodel.load_state_dict(torch.load(os.path.join(BASE_DIR, "Trainig/state_dictmodel.pth")))
        
        # Make prediction
        loadedmodel.eval()
        with torch.no_grad():
            y_pred = loadedmodel(input_tensor_scaled)
            y_pred_raw = torch.exp(y_pred * Y_std + Y_mean)  # Reverse log transformation
            print(f'Prediction: {y_pred_raw.item():.1f}')  
                
        # Return the prediction
        return y_pred_raw.item()

@app.post("/predict")
def predict(property: Property):
    encoded = []
    
    encoded.extend([property.Bedrooms, property.Bathrooms, property.Land_Area, property.Year])

    status_encoder = Encode("Preprocessing/Statusonehot_encoder.pkl", property.Status, "Status")
    status_encoder.encode()
    encoded.extend(status_encoder.get_encoded_data().flatten().tolist())

    furnished_encoder = Encode("Preprocessing/Furnishedonehot_encoder.pkl", property.Furnished, "Furnished")
    furnished_encoder.encode()
    encoded.extend(furnished_encoder.get_encoded_data().flatten().tolist())

    address_encoder = Encode("Preprocessing/Addressonehot_encoder.pkl", property.Address, "Address")
    address_encoder.encode()
    encoded.extend(address_encoder.get_encoded_data().flatten().tolist())

    property_type_encoder = Encode("Preprocessing/Property_Typeonehot_encoder.pkl", property.Property_Type, "Property_Type")
    property_type_encoder.encode()
    encoded.extend(property_type_encoder.get_encoded_data().flatten().tolist())


    predictedprice=predictPrice(encoded)

    return {"Predicted Price": predictedprice}

