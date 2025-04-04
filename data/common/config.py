import json

class Config:
    def __init__(self, config_file_path):
        """
        Initialize the Config class with the path to the config.json file.

        Parameters:
        -----------
        config_file_path : str
            Path to the config.json file.
        """
        self.config_file_path = config_file_path
        self.config_data = self._load_config()

    def _load_config(self):
        """
        Load the configuration from the JSON file.

        Returns:
        --------
        dict : The configuration data.
        """
        try:
            with open(self.config_file_path, 'r') as file:
                config_data = json.load(file)
            print("Configuration loaded successfully.")
            return config_data
        except Exception as e:
            print(f"An error occurred while loading the configuration: {e}")
            return {}

    def get_model_name(self):
        """
        Get the model name from the configuration.

        Returns:
        --------
        str : The model name.
        """
        return self.config_data.get('model_name', '')

    def get_model_work_path(self):
        """
        Get the model work path from the configuration.

        Returns:
        --------
        str : The model work path.
        """
        return self.config_data.get('model_work_path', '')

    def get_dictionary_path(self):
        """
        Get the dictionary path from the configuration.

        Returns:
        --------
        str : The dictionary path.
        """
        return self.config_data.get('dictionary_path', '')

    def get_segments(self):
        """
        Get the list of segments from the configuration.

        Returns:
        --------
        list : A list of segments, each containing data and output paths.
        """
        return self.config_data.get('segments', [])

    def get_segment_data(self, segment_name):
        """
        Get the data path for a specific segment.

        Parameters:
        -----------
        segment_name : str
            The name of the segment.

        Returns:
        --------
        str : The data path for the segment, or an empty string if not found.
        """
        for segment in self.get_segments():
            if segment.get('name') == segment_name:
                return segment.get('data', '')
        return ''

    def get_segment_output(self, segment_name):
        """
        Get the output path for a specific segment.

        Parameters:
        -----------
        segment_name : str
            The name of the segment.

        Returns:
        --------
        str : The output path for the segment, or an empty string if not found.
        """
        for segment in self.get_segments():
            if segment.get('name') == segment_name:
                return segment.get('output', '')
        return ''
