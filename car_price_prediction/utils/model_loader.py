import os
import pickle

class ModelLoader:

    @classmethod
    def get_latest_file_path(cls, directory: str, extension: str) -> str:
        model_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension) and f.startswith('car_price_model')]
        column_names_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl') and f.startswith('column_names')]
        if not model_files:
            raise FileNotFoundError(f"No files found in the specified directory for extension {extension}.")
        latest_model_file = max(model_files, key=os.path.getmtime)
        latest_column_names_file = max(column_names_files, key=os.path.getmtime)
        # Load the model
        with open(latest_model_file, 'rb') as model_file:
            model = pickle.load(model_file)

        # Load the column names
        with open(latest_column_names_file, 'rb') as col_file:
            column_names = pickle.load(col_file)
                
        return model, column_names