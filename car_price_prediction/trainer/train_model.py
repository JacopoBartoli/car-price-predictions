from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from car_price_prediction.preprocessing.carDataManager import CarDataManager

class CarPricePredictor:
    """
    A class to predict car prices using a Random Forest model.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset CSV file.

    Attributes
    ----------
    model : RandomForestRegressor
        The Random Forest model used for predictions.
    X : DataFrame
        The features used for training the model.
    y : Series
        The target variable (car prices).
    data_manager : CarDataManager
        An instance of CarDataManager to handle data loading and preprocessing.

    Methods
    -------
    train()
        Trains the Random Forest model on the loaded data.
    
    save_model(model_path: str)
        Saves the trained model and column names to disk.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.model = RandomForestRegressor()
        self.data_manager = CarDataManager(dataset_path)

    def train(self):
        self.data_manager.load_data()
        self.X = self.data_manager.X
        self.y = self.data_manager.y
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        print(f"Model R^2 Score: {score}")

    def save_model(self, model_path: str, column_names_path: str):
        # Save the model
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

        # Save column names
        with open(column_names_path, 'wb') as col_file:
            pickle.dump(self.X.columns.tolist(), col_file)

if __name__ == "__main__":
    dataset_path = "./data/used_cars.csv"
    model_path = f"./models/car_price_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    column_names_path = f"./models/column_names_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    predictor = CarPricePredictor(dataset_path)
    predictor.train()
    predictor.save_model(model_path, column_names_path)

