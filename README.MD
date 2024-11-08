# Car Price Prediction API

## Overview

This project is a FastAPI application that predicts car prices using a machine learning model based on a dataset of used cars. The application exposes an API endpoint for making predictions and integrates with Prometheus for monitoring metrics.

## Project Goal
The goal of this project is to create a FastAPI application and integrate it with Prometheus/Grafana. The data exploration, preprocessing, and machine learning components of this project are very simple and not fully developed, as they are not the main focus of the project. In addition, given the rudimentary nature of the FastAPI application, the structural framework is not as refined as it could be.

## Dataset
The dataset is the public dataset [Used Car Prices Prediction](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).

## Features

- **Car Price Prediction**: Predicts the price of a car based on various features such as model year, mileage, brand, model, fuel type, and transmission.
- **Metrics Monitoring**: Integrates with Prometheus to expose metrics for monitoring the application's performance and usage.
- **Multi-Pod Support**: Designed to work in a multi-pod environment, allowing for horizontal scaling.

## Technologies Used

- **FastAPI**: A modern web framework for building APIs with Python 3.6+ based on standard Python type hints.
- **Scikit-learn**: A machine learning library for Python that provides simple and efficient tools for data mining and data analysis.
- **Pandas**: A data manipulation and analysis library for Python.
- **Prometheus**: An open-source monitoring and alerting toolkit.

## Project Structure

```
.
├── data/                 
├── model/
├── configuration/                 
├── car_price_prediction/                   
│   ├── api_testing/
│   │   └── api_testing.py
│   ├── etc/
│   │   └── prometheus.py/
│   │       └── prometheus.yaml
│   ├── preprocessing/
│   │   ├── carDataManager.py
│   │   └── inputPreprocessor.py/
│   ├── trainer/
│   │   └── train_model.py/
│   ├── utils/
│   │   ├── config_reader.py
│   │   └── model_loader.py
│   └── app.py/
├── configuration/
│   └── config.ini
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Used Car Prices Prediction](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).

4. Place your dataset in the `data/` directory.

5. Place your trained model in the `model/` directory. If you don't have a trained model already, there is an easy script in the `car_price_prediction/trainer` folder to train a simple model.

## Running the Application

You can run the application using Docker and Docker Compose:

1. Build and start the services:

   ```bash
   docker-compose up --build
   ```

2. Access the FastAPI application at `http://localhost:8000`.

3. Access the Prometheus monitoring interface at `http://localhost:9090`.

4. Access the Grafana monitoring interface at `http://localhost:3000`.

## API Endpoints

### Predict Car Price

- **Endpoint**: `/predict/`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "model_year": 2020,
    "mileage": 15000,
    "brand": "Toyota",
    "fuel_type": "Gasoline",
    "accident": "No",
    "clean_title": "Yes",
    "ext_col": "Black",
    "int_col": "Black"
  }
  ```

- **Response**:
  ```json
  {
    "predicted_price": 25000.0
  }
  ```

## Prometheus Configuration

The Prometheus configuration file is located at `configuration/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['app:8000']
  
  - job_name: 'prometheus'
    static_configs:
      - targets: ['prometheus:9090']

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['node_exporter:9100']
      
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### Prometheus Exporters
In this project, I used Prometheus third-party exporters to help export existing metrics from third-party systems as Prometheus metrics. In particular, I used:
- Node/system metrics exporter
- cAdvisor

A complete list of third-party exporters can be found [here](https://prometheus.io/docs/instrumenting/exporters/#third-party-exporters).

## Grafana Suggested Dashboards
You can easily export some dashboards from the Grafana dashboard website. For this project, I suggest:
1. [Node Exporter](https://grafana.com/grafana/dashboards/1860-node-exporter-full/)

2. [cAdvisor Exporter](https://grafana.com/grafana/dashboards/14282-cadvisor-exporter/)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.