import configparser
import os

class Config:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.settings = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        # Convert the config to a dictionary
        settings = {section: dict(config.items(section)) for section in config.sections()}
        return settings

    def get(self, section: str, key: str, default=None):
        return self.settings.get(section, {}).get(key, default)
    
configs = Config("./configuration/config.ini")