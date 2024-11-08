import requests
import random

# API address
API_URL = "http://localhost:8000"

car_data = {
    "model_year": 2020,
    "mileage": 15000,
    "brand": "Toyota",
    "fuel_type": "Gasoline",
    "accident": "No",
    "clean_title": "Yes",
    "ext_col": "Black",
    "int_col": "Black"
}

brands = ["Toyota", "Ford", "BMW", "Audi", "Honda"]
model_years = [2018, 2019, 2020, 2021, 2022]
mileage_options = [10000, 20000, 30000, 40000, 50000]

def randomize_car_data():
    return {
        "model_year": random.choice(model_years),
        "mileage": random.choice(mileage_options),
        "brand": random.choice(brands),
        "fuel_type": "Gasoline",
        "accident": "No",
        "clean_title": "Yes",
        "ext_col": "Black",
        "int_col": "Black"
    }

def test_prediction(num_requests=5):
    for _ in range(num_requests):
        car_data = randomize_car_data()
        response = requests.post(f"{API_URL}/predict/", json=car_data)
        if response.status_code == 200:
            print("Prediction:", response.json())
        else:
            print("Prediction error:", response.status_code, response.text)

def test_metrics():
    response = requests.get(f"{API_URL}/metrics")
    if response.status_code == 200:
        print("Metrics obtained:")
        print(response.text)
    else:
        print("Error in metrics retrieval:", response.status_code, response.text)

if __name__ == "__main__":
    test_prediction(num_requests=500)
    test_metrics()
