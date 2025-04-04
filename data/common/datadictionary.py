import pandas as pd

class DataDictionary:
    def __init__(self, excel_file_path):
        """
        Initialize the DataDictionary with the path to the Excel file.

        Parameters:
        -----------
        excel_file_path : str
            Path to the Excel file containing the data dictionary.
        """
        self.excel_file_path = excel_file_path
        self.data_dict = None
        self.generated_dictionary = None

    def load_data_dictionary(self, sheet_name=0):
        """
        Load the data dictionary from the Excel file.

        Parameters:
        -----------
        sheet_name : str or int, default 0
            Name or index of the sheet to read from the Excel file.
        """
        try:
            self.data_dict = pd.read_excel(self.excel_file_path, sheet_name=sheet_name)
            print("Data dictionary loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the data dictionary: {e}")

    def get_variable_description(self, variable_name):
        """
        Get the description of a specific variable.

        Parameters:
        -----------
        variable_name : str
            The name of the variable to get the description for.

        Returns:
        --------
        str or None : The description of the variable, or None if not found.
        """
        if self.data_dict is not None:
            description = self.data_dict.loc[self.data_dict['Variable'] == variable_name, 'Description']
            if not description.empty:
                return description.iloc[0]
            else:
                print(f"Variable '{variable_name}' not found in the data dictionary.")
        else:
            print("Data dictionary is not loaded.")
        return None

    def get_all_variables(self):
        """
        Get a list of all variables in the data dictionary.

        Returns:
        --------
        list : A list of variable names.
        """
        if self.data_dict is not None:
            return self.data_dict['Variable'].tolist()
        else:
            print("Data dictionary is not loaded.")
            return []

    def display_data_dictionary(self):
        """
        Display the entire data dictionary.
        """
        if self.data_dict is not None:
            print(self.data_dict)
        else:
            print("Data dictionary is not loaded.")

    def generate_dataframe(self, input_df, exclude_columns=None):
        """
        Generate a pandas DataFrame with 'Variable', 'Description', 'Type', 'Remove', and 'Remove Reason' columns.

        Parameters:
        -----------
        input_df : pandas DataFrame
            The input DataFrame to analyze.
        exclude_columns : list, default None
            List of column names to exclude from the DataFrame.

        Returns:
        --------
        pandas DataFrame : DataFrame with 'Variable', 'Description', 'Type', 'Remove', and 'Remove Reason' columns.
        """
        # Prepare the DataFrame structure
        variables = input_df.columns.difference(exclude_columns or [])
        result_df = pd.DataFrame(variables, columns=['Variable'])

        # Join with self.data_dict to get descriptions if available
        if self.data_dict is not None:
            result_df = result_df.merge(self.data_dict, on='Variable', how='left')
        else:
            result_df['Description'] = None

        # Detect variable types
        result_df['Type'] = result_df['Variable'].apply(lambda var: self.detect_type(input_df[var]))

        # Initialize 'Remove' and 'Remove Reason' columns
        result_df['Remove'] = 0
        result_df['Remove Reason'] = ""

        self.generated_dictionary = result_df

    def update_removal_status(self, variables_to_remove, reason):
        """
        Update the 'Remove' and 'Remove Reason' columns for specified variables.

        Parameters:
        -----------
        variables_to_remove : list
            List of variable names to mark for removal.
        reason : str
            The reason for marking these variables for removal.
        """
        if self.generated_dictionary is None:
            print("Generated dictionary is not available.")
            return

        # Update the 'Remove' and 'Remove Reason' columns
        self.generated_dictionary.loc[self.generated_dictionary['Variable'].isin(variables_to_remove), 'Remove'] = 1
        self.generated_dictionary.loc[self.generated_dictionary['Variable'].isin(variables_to_remove), 'Remove Reason'] = reason

    def append_null_statistics(self, input_df):
        """
        Append null statistics to the generated dictionary.

        Parameters:
        -----------
        input_df : pandas DataFrame
            The input DataFrame to analyze.

        Modifies:
        ---------
        Updates the generated_dictionary with null statistics.
        """
        if self.generated_dictionary is None:
            print("Generated dictionary is not available.")
            return

        # Calculate null statistics
        null_counts = input_df.isnull().sum()
        total_counts = len(input_df)
        null_percentages = (null_counts / total_counts) * 100

        # Append statistics to the generated dictionary
        self.generated_dictionary['Null Count'] = self.generated_dictionary['Variable'].map(null_counts)
        self.generated_dictionary['Null Percentage'] = self.generated_dictionary['Variable'].map(null_percentages)
        self.generated_dictionary['Total Count'] = total_counts

    def append_column_types(self, input_df):
        """
        Append the column data types of the input DataFrame to the generated dictionary.

        Parameters:
        -----------
        input_df : pandas DataFrame
            The input DataFrame to analyze.

        Modifies:
        ---------
        Updates the generated_dictionary with column data types.
        """
        if self.generated_dictionary is None:
            print("Generated dictionary is not available.")
            return

        # Get data types of each column
        column_types = input_df.dtypes

        # Append data types to the generated dictionary
        self.generated_dictionary['Data Type'] = self.generated_dictionary['Variable'].map(column_types)

    def append_additional_statistics(self, input_df):
        """
        Append cardinality, freqmax, concentration, mode, max, min, mean, std, and percentiles to the generated dictionary.

        Parameters:
        -----------
        input_df : pandas DataFrame
            The input DataFrame to analyze.

        Modifies:
        ---------
        Updates the generated_dictionary with additional statistics.
        """
        if self.generated_dictionary is None:
            print("Generated dictionary is not available.")
            return

        # Calculate additional statistics
        cardinality = input_df.nunique()
        mode = input_df.mode().iloc[0]
        freqmax = input_df.apply(lambda x: x.value_counts().max())
        concentration = (freqmax / len(input_df)) * 100
        max_values = input_df.max()
        min_values = input_df.min()
        mean_values = input_df.mean()
        std_values = input_df.std()
        percentiles_25 = input_df.quantile(0.25)
        percentiles_50 = input_df.quantile(0.50)
        percentiles_75 = input_df.quantile(0.75)

        # Append statistics to the generated dictionary
        self.generated_dictionary['Cardinality'] = self.generated_dictionary['Variable'].map(cardinality)
        self.generated_dictionary['Mode'] = self.generated_dictionary['Variable'].map(mode)
        self.generated_dictionary['FreqMax'] = self.generated_dictionary['Variable'].map(freqmax)
        self.generated_dictionary['Concentration'] = self.generated_dictionary['Variable'].map(concentration)
        self.generated_dictionary['Max'] = self.generated_dictionary['Variable'].map(max_values)
        self.generated_dictionary['Min'] = self.generated_dictionary['Variable'].map(min_values)
        self.generated_dictionary['Mean'] = self.generated_dictionary['Variable'].map(mean_values)
        self.generated_dictionary['Std'] = self.generated_dictionary['Variable'].map(std_values)
        self.generated_dictionary['25%'] = self.generated_dictionary['Variable'].map(percentiles_25)
        self.generated_dictionary['50%'] = self.generated_dictionary['Variable'].map(percentiles_50)
        self.generated_dictionary['75%'] = self.generated_dictionary['Variable'].map(percentiles_75)

    @staticmethod
    def detect_type(series):
        """
        Detect the type of a variable based on its data.

        Parameters:
        -----------
        series : pandas Series
            The data series to analyze.

        Returns:
        --------
        str : The type of the variable ('Ratio', 'Discrete', 'Nominal').
        """
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 10:
                return 'Discrete'
            else:
                return 'Ratio'
        else:
            return 'Nominal'