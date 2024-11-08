import pandas as pd

class InputPreprocessor:
    """
    Class for preprocessing input data for car price prediction.

    Parameters
    ----------
    column_names : list[str]
        A list of column names to be used in the output DataFrame.

    Methods
    -------
    preprocess(input_data: dict) -> list[float]:
        Preprocesses the input data and returns a list of float values.
    """

    def __init__(self, column_names: list[str]):
        """
        Initializes the InputPreprocessor object.

        Parameters
        ----------
        column_names : list[str]
            A list of column names to be used in the output DataFrame.
        """
        self.column_names = column_names

    def preprocess(self, input_data: dict) -> list[float]:
        """
        Preprocesses the input data and returns a list of float values.

        Parameters
        ----------
        input_data : dict
            A dictionary containing the car data to preprocess. It must contain the following keys:
            - model_year
            - mileage
            - brand
            - fuel_type
            - ext_col
            - int_col
            - accident
            - clean_title

        Returns
        -------
        list[float]
            A list of float values representing the preprocessed data.
        """
        input_data = {
            'model_year':  input_data.model_year,
            'mileage': input_data.mileage,
            'brand': input_data.brand,
            'fuel_type': input_data.fuel_type,
            'ext_col': input_data.ext_col,
            'int_col': input_data.int_col,
            'accident': 1 if input_data.accident == "Yes" else 0,
            'clean_title': 1 if input_data.clean_title == "Yes" else 0
        }
        
        out_df = pd.DataFrame([input_data])
        out_df = pd.get_dummies(out_df, columns=['brand', 'fuel_type', 'ext_col', 'int_col'], drop_first=True)
        
        missing_cols = set(self.column_names) - set(out_df.columns)

        missing_data = {col: 0 for col in missing_cols}

        out_df = pd.concat([out_df, pd.DataFrame([missing_data])], axis=1)
        
        out_df = out_df[self.column_names]

        return out_df
    