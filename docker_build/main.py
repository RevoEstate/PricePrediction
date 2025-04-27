from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import torch
from model import RevoNeuralNetwork
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
app = FastAPI(title="RevoEstate Price Prediction API", version="1.0.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Property(BaseModel):
    Bedrooms: int = Field(ge=0,lt=100, description="Number of bedrooms")
    Bathrooms: int = Field(ge=0,lt=100, description="Number of bathrooms")
    Land_Area: float = Field(ge=0.0,lt=10000, description="Land area in square units")
    Year: int = Field(ge=2023, le=2030, description="Year of construction")
    Status: Literal['For Rent', 'For Sale']
    Furnished: Literal['Yes', 'No']
    Address: Literal['Addis Ketema', 'Akaky Kaliti', 'Arada', 'Bole', 'Gullele', 'Kirkos', 
                     'Kolfe Keranio', 'Lideta', 'Nifas Silk-Lafto', 'Yeka', 'Lemi Kura']
    Property_Type: Literal['Apartment', 'Villa']

class Encode:
    def __init__(self, filename: str, value: str, column_name: str):
        self.filename = filename  # No BASE_DIR, files are in /app/
        self.value = value
        self.column_name = column_name
        self.encoded_data = None

    def encode(self):
        try:
            print(f"Loading {self.filename}")
            self.encoder = joblib.load(self.filename)
            print(f"Encoder categories: {self.encoder.categories_}")
            input_data = pd.DataFrame([[self.value]], columns=[self.column_name])
            print(f"Input: {input_data}")
            self.encoded_data = self.encoder.transform(input_data)
            print(f"Encoded: {self.encoded_data}")
       
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
        mean = torch.load("mean.pt")
        std = torch.load("std.pt")
        Y_mean = torch.load("Y_mean.pt")
        Y_std = torch.load("Y_std.pt")

        input_tensor = torch.tensor(encoded_data, dtype=torch.float32).unsqueeze(0)
        input_tensor_scaled = (input_tensor - mean) / std

        model = RevoNeuralNetwork()
        model.load_state_dict(torch.load("state_dictmodel.pth"))
        model.eval()

        with torch.no_grad():
            y_pred = model(input_tensor_scaled)
            y_pred_raw = torch.exp(y_pred * Y_std + Y_mean)
            predicted_price = y_pred_raw.item()
            if predicted_price < 0:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unexpected error occurred during prediction"
                )
            return predicted_price
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during prediction"
        )
@app.get("/", response_model=dict)
@limiter.limit("100/hour")
async def welcome_to_revoestate_priceprediction(request: Request):
    base_url = str(request.base_url)
    return{
  "message": "Hello! Welcome to RevoEstate Price Prediction API",
  "personal_note": "Crafted to make property price predictions simple and reliable!",
  "description": "This API predicts property prices based on features like bedrooms, bathrooms, land area, and location.",
  "documentation": "https://jibrla-revoestate.hf.space/docs",
  "documentation_note": "Paste this URL in your browser for interactive API docs (Swagger UI).",
  "alternative_docs": "https://jibrla-revoestate.hf.space/redoc",
  "alternative_docs_note": "Paste this URL for the ReDoc interface.",
  "version": "1.0.0",
  "test_predict": "Send a POST request to /predict to get a price prediction. Example using curl:\n curl -X POST https://jibrla-revoestate.hf.space/predict -H 'Content-Type: application/json' -d '{\"Bedrooms\": 3, \"Bathrooms\": 2, \"Land_Area\": 100.0, \"Year\": 2023, \"Status\": \"For Sale\", \"Furnished\": \"Yes\", \"Address\": \"Bole\", \"Property_Type\": \"Apartment\"}'",
  "quick_tip": "Visit /docs to try the API directly in your browser!"
}


@app.post("/predict", response_model=dict)
@limiter.limit("100/hour")  
async def predict(property: Property, request: Request):
    try:
        if property.Land_Area <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Land area must be greater than zero"
            )

        encoded = [
            property.Bedrooms,
            property.Bathrooms,
            property.Land_Area,
            property.Year
        ]

        encoders = [
            ("Status", "Statusonehot_encoder.pkl", property.Status),
            ("Furnished", "Furnishedonehot_encoder.pkl", property.Furnished),
            ("Address", "Addressonehot_encoder.pkl", property.Address),
            ("Property_Type", "Property_Typeonehot_encoder.pkl", property.Property_Type)
        ]

        for column_name, file_path, value in encoders:
            encoder = Encode(file_path, value, column_name)
            encoder.encode()
            encoded.extend(encoder.get_encoded_data().flatten().tolist())

        predicted_price = predictPrice(encoded)

        return {"PredictedPrice": predicted_price}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction" 
        )
# Custom 404 handler
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Endpoint not found: {request.url.path}"},
    )
