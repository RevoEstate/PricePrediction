from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import os
import torch
from model import RevoNeuralNetwork



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
    Bedrooms: int = Field(ge=0, description="Number of bedrooms")
    Bathrooms: int = Field(ge=0, description="Number of bathrooms")
    Land_Area: float = Field(ge=0.0, description="Land area in square units")
    Year: int = Field(ge=2023, le=2030, description="Year of construction")
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
        try:
            self.encoder = joblib.load(self.filename)
            input_data = pd.DataFrame([[self.value]], columns=[self.column_name])
            self.encoded_data = self.encoder.transform(input_data)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Encoder file not found: {self.filename}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to encode {self.column_name}"
            )

    def get_encoded_data(self):
        if self.encoded_data is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Encoding not performed"
            )
        return self.encoded_data

def predictPrice(encoded_data):
    try:
        # Load scaling parameters
        mean = torch.load(os.path.join(BASE_DIR, "Trainig/mean.pt"))
        std = torch.load(os.path.join(BASE_DIR, "Trainig/std.pt"))
        Y_mean = torch.load(os.path.join(BASE_DIR, "Trainig/Y_mean.pt"))
        Y_std = torch.load(os.path.join(BASE_DIR, "Trainig/Y_std.pt"))

        # Convert input to tensor
        input_tensor = torch.tensor(encoded_data, dtype=torch.float32).unsqueeze(0)
        input_tensor_scaled = (input_tensor - mean) / std

        # Load the model
        model = RevoNeuralNetwork()
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, "Trainig/state_dictmodel.pth")))
        model.eval()

        # Make prediction
        with torch.no_grad():
            y_pred = model(input_tensor_scaled)
            y_pred_raw = torch.exp(y_pred * Y_std + Y_mean)  # Reverse log transformation
            predicted_price = y_pred_raw.item()
            if predicted_price < 0:
                raise  HTTPException(
                       status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                       detail="An unexpected error occurred during prediction"
        )
            return predicted_price

    # except FileNotFoundError as e:
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail="Model or scaling files not found"
    #     )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during prediction"
        )

@app.post("/predict")
async def predict(property: Property):
    try:
        # Validate input data
        if property.Land_Area <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Land area must be greater than zero"
            )

        # Encode features
        encoded = [
            property.Bedrooms,
            property.Bathrooms,
            property.Land_Area,
            property.Year
        ]

        # Encode categorical features
        encoders = [
            ("Status", "Preprocessing/Statusonehot_encoder.pkl", property.Status),
            ("Furnished", "Preprocessing/Furnishedonehot_encoder.pkl", property.Furnished),
            ("Address", "Preprocessing/Addressonehot_encoder.pkl", property.Address),
            ("Property_Type", "Preprocessing/Property_Typeonehot_encoder.pkl", property.Property_Type)
        ]

        for column_name, file_path, value in encoders:
            encoder = Encode(file_path, value, column_name)
            encoder.encode()
            encoded.extend(encoder.get_encoded_data().flatten().tolist())

        # Make prediction
        predicted_price = predictPrice(encoded)

        return {"Predicted Price": predicted_price}

    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction"
        )