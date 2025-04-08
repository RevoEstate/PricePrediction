from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import os

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

    return {"encoded_features": encoded}

