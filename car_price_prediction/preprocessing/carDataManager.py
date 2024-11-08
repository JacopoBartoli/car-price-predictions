import pandas as pd
import re

class CarDataManager:
    """
    A class to manage car data loading and preprocessing.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset CSV file.

    Attributes
    ----------
    data : DataFrame
        The DataFrame containing the loaded and processed data.
    X : DataFrame
        The features used for training the model.
    y : Series
        The target variable (car prices).

    Methods
    -------
    load_data() -> None:
        Loads the dataset from the specified path and preprocesses it.
    
    convert_mileage(mileage: str) -> float:
        Converts mileage from a formatted string to a float.
    """

    def __init__(self, dataset_path: str):
        """
        Initializes the CarDataManager with the dataset path.

        Parameters
        ----------
        dataset_path : str
            The path to the dataset CSV file.
        """
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None

    def load_data(self) -> None:
        """
        Loads the dataset from the specified path and preprocesses it.

        This method performs the following operations:
        - Loads the dataset into a DataFrame.
        - Converts mileage from a formatted string to a float.
        - Drops unnecessary columns.
        - Encodes categorical variables.
        - Checks for the presence of the 'price' column and separates features and target variable.

        Raises
        ------
        KeyError
            If the 'price' column is not present in the DataFrame.
        """
        self.data = pd.read_csv(self.dataset_path)
        
        # Convert mileage
        self.data['mileage'] = self.data['milage'].apply(self.convert_mileage)
        self.data.drop(columns=['milage'], inplace=True)

        # Remove unnecessary columns
        self.data.drop(columns=['engine', 'transmission', 'model'], inplace=True, errors='ignore')

        # Replace non-numeric values in the 'accident' column
        self.data['accident'] = self.data['accident'].replace({'At least 1 accident or damage reported': 1, 'No accident': 0})
        self.data['accident'] = pd.to_numeric(self.data['accident'], errors='coerce')  # Convert to numeric, replace errors with NaN
        self.data.dropna(inplace=True)  # Remove rows with NaN

        # Convert 'price' column to float
        self.data['price'] = self.data['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

        # Encode categorical variables
        self.data = pd.get_dummies(self.data, columns=['brand', 'fuel_type', 'ext_col', 'int_col', 'clean_title'], drop_first=True)

        # Check if 'price' column is present
        if 'price' not in self.data.columns:
            raise KeyError("The 'price' column is not present in the DataFrame.")

        self.X = self.data.drop(columns=['price'])
        self.y = self.data['price']

    def convert_mileage(self, mileage: str) -> float:
        """
        Converts mileage from a formatted string to a float.

        Parameters
        ----------
        mileage : str
            The mileage string to convert (e.g., "51,000 mi.").

        Returns
        -------
        float
            The converted mileage as a float.
        """
        return float(re.sub(r'[^\d]', '', mileage))