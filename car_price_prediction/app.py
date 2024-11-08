import os
from pydantic import BaseModel
from fastapi import FastAPI, APIRouter, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram,CollectorRegistry, generate_latest
from prometheus_client import multiprocess

from car_price_prediction.utils.model_loader import ModelLoader
from car_price_prediction.utils.config_reader import configs
from car_price_prediction.preprocessing.inputPreprocessor import InputPreprocessor

# Define the input data model
class CarFeatures(BaseModel):
    model_year: int
    mileage: int
    brand: str
    ext_col: str
    int_col: str
    fuel_type: str
    accident: str
    clean_title: str

# Define the out response model
class PredictionResponse(BaseModel):
    predicted_price: float

# Create the FastAPI app and router
app = FastAPI()
router = APIRouter()

app.add_middleware(CORSMiddleware, 
                   allow_origins=["*"])

# Get configuration paths
BASE_DIRECTORY = configs.get("paths", 'base_directory')
MODELS_DIRECTORY = os.path.join(BASE_DIRECTORY, configs.get("paths", 'models_directory'))
PROMETHEUS_DIRECTORY = os.path.join(BASE_DIRECTORY, configs.get("paths", 'prometheus_directory'))

if not os.path.exists(PROMETHEUS_DIRECTORY):
    os.makedirs(PROMETHEUS_DIRECTORY)
    print(f"Created directory: {PROMETHEUS_DIRECTORY}")
else:
    print(f"Using existing directory: {PROMETHEUS_DIRECTORY}")
# Set the Prometheus multiprocess directory
os.environ['PROMETHEUS_MULTIPROC_DIR'] = PROMETHEUS_DIRECTORY

# Load the latest model and column names
model, column_names = ModelLoader.get_latest_file_path(MODELS_DIRECTORY, '.pkl')

# Create the Prometheus registry
registry = CollectorRegistry()

#Create the Prometheus metrics
prediction_counter = Counter('prediction_requests', 'The total number of prediction requests', registry=registry)
brand_prediction_counter = Counter('brand_prediction_requests', 'The number of car predictions for each brand', ['brand'], registry=registry)
price_counter = Histogram('price_counter', 'The mean prediction counter', registry=registry)

@router.post("/predict/", response_model=PredictionResponse)
def predict(car: CarFeatures):
    prediction_counter.inc()
    brand_prediction_counter.labels(brand=car.brand).inc()
    
    input_data = [[
        car.model_year,
        car.mileage,
        car.brand,
        car.ext_col,
        car.int_col,
        car.fuel_type,
        car.accident,
        car.clean_title
    ]]
    
    input_preprocessing = InputPreprocessor(column_names)
    input_df = input_preprocessing.preprocess(car)

    predicted_price = model.predict(input_df)[0]
    price_counter.observe(predicted_price)
    return PredictionResponse(predicted_price=predicted_price)

@app.get("/metrics")
def metrics():
    multiprocess.MultiProcessCollector(registry)
    return Response(content=generate_latest(registry), media_type="text/plain")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    host = configs.get("urls", "api_host")
    port = configs.get("urls", "api_port")
    uvicorn.run(app, host=host, port=port)