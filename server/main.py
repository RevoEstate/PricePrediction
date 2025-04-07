from fastapi import FastAPI
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Literal,Annotated

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
    Land_Area:float
    Year: int
    Status: Literal['For Rent', 'For Sale']
    Furnished: Literal['Yes', 'No']
    Address:Literal['Addis Ketema', 'Akaky Kaliti', 'Arada', 'Bole', 'Gullele', 'Kirkos', 'Kolfe Keranio', 'Lideta', 'Nifas Silk- Lafto', 'Yeka','Lemi Kura' ]
    Property_Type:Literal['apartment', 'villa']
@app.get("/predict")
def predict( property: Property):
    pass
   