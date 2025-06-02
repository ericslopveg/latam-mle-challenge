import fastapi
import pandas as pd
from pydantic import BaseModel, validator, ValidationError
from typing import List
from fastapi import Request
from fastapi.responses import JSONResponse
from challenge.model import DelayModel

app = fastapi.FastAPI()

# Custom exception handler to convert 422 to 400
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Validation error"}
    )

@app.exception_handler(fastapi.exceptions.RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: fastapi.exceptions.RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Validation error"}
    )

# Load and train the model once when the API starts
model = DelayModel()

# Load training data and train the model
try:
    # Try to load the data and train the model
    data = pd.read_csv("data/data.csv")
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
except Exception as e:
    print(f"Warning: Could not train model on startup: {e}")
    # Model will use dummy training when predict is called


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    
    @validator('MES')
    def validate_mes(cls, v):
        if not (1 <= v <= 12):
            raise ValueError('MES must be between 1 and 12')
        return v
    
    @validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in ['N', 'I']:
            raise ValueError('TIPOVUELO must be N or I')
        return v
    
    @validator('OPERA')
    def validate_opera(cls, v):
        # Valid airline operators based on the model's training data
        valid_operators = [
            'Grupo LATAM', 'Sky Airline', 'Aerolineas Argentinas', 'Copa Air',
            'Latin American Wings', 'Avianca', 'JetSmart SPA', 'Gol Trans',
            'American Airlines', 'Air Canada', 'Iberia', 'Delta Air',
            'Air France', 'Alitalia', 'KLM', 'British Airways', 'Qantas Airways',
            'United Airlines', 'Lacsa', 'Austral', 'Plus Ultra Lineas Aereas'
        ]
        if v not in valid_operators:
            raise ValueError(f'OPERA must be one of: {valid_operators}')
        return v


class FlightData(BaseModel):
    flights: List[Flight]


class PredictionResponse(BaseModel):
    predict: List[int]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(flight_data: FlightData) -> PredictionResponse:
    try:
        # Convert flights to DataFrame
        flights_list = []
        for flight in flight_data.flights:
            flights_list.append({
                'OPERA': flight.OPERA,
                'TIPOVUELO': flight.TIPOVUELO,
                'MES': flight.MES,
                # Add required columns for preprocessing
                'Fecha-I': '2023-01-01 10:00:00'  # Dummy date for preprocessing
            })
        
        flights_df = pd.DataFrame(flights_list)
        
        # Preprocess the data
        features = model.preprocess(flights_df)
        
        # Make predictions
        predictions = model.predict(features)
        
        return PredictionResponse(predict=predictions)
        
    except ValueError as e:
        # Return 400 for validation errors
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Return 500 for other errors
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")