import pandas as pd
import numpy as np

class DataDictionary:
    def __init__(self, default_dictionary_location=None):
        """
        Initialize the DataDictionary with the path to the Excel file.

        Parameters:
        -----------
        excel_file_path : str
            Path to the Excel file containing the data dictionary.
        """

        self.data_dict = None
        self.generated_dictionary = None
        self.default_dictionary_location = default_dictionary_location

    def load_data_dictionary(self, excel_file_path, sheet_name=0):
        """
        Load the data dictionary from the Excel file.

        Parameters:
        -----------
        sheet_name : str or int, default 0
            Name or index of the sheet to read from the Excel file.
        """
        self.excel_file_path = excel_file_path
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
       
        self.append_column_types(input_df)
        self.append_null_statistics(input_df)
        self.append_additional_statistics(input_df)
        

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
        """
        if self.generated_dictionary is None:
            print("Generated dictionary is not available.")
            return

        # Identify numeric variables
        numeric_vars = input_df.select_dtypes(include=np.number).columns
        numeric_mask = self.generated_dictionary['Variable'].isin(numeric_vars)
        
        # Calculate additional statistics
        cardinality = input_df.nunique()
        mode = input_df.mode().iloc[0]
        freqmax = input_df.apply(lambda x: x.value_counts().max())
        concentration = (freqmax / len(input_df)) * 100
        
        # Initialize numeric stats with NaN
        max_values = pd.Series(np.nan, index=input_df.columns)
        min_values = pd.Series(np.nan, index=input_df.columns)
        mean_values = pd.Series(np.nan, index=input_df.columns)
        std_values = pd.Series(np.nan, index=input_df.columns)
        percentiles_25 = pd.Series(np.nan, index=input_df.columns)
        percentiles_50 = pd.Series(np.nan, index=input_df.columns)
        percentiles_75 = pd.Series(np.nan, index=input_df.columns)
        
        # Calculate only for numeric variables
        max_values[numeric_vars] = input_df[numeric_vars].max()
        min_values[numeric_vars] = input_df[numeric_vars].min()
        mean_values[numeric_vars] = input_df[numeric_vars].mean()
        std_values[numeric_vars] = input_df[numeric_vars].std()
        percentiles_25[numeric_vars] = input_df[numeric_vars].quantile(0.25)
        percentiles_50[numeric_vars] = input_df[numeric_vars].quantile(0.50)
        percentiles_75[numeric_vars] = input_df[numeric_vars].quantile(0.75)

        # Append statistics to the generated dictionary
        self.generated_dictionary['Cardinality'] = self.generated_dictionary['Variable'].map(cardinality)
        self.generated_dictionary['Mode'] = self.generated_dictionary['Variable'].map(mode)
        self.generated_dictionary['FreqMax'] = self.generated_dictionary['Variable'].map(freqmax)
        self.generated_dictionary['Concentration'] = self.generated_dictionary['Variable'].map(concentration)
        
        # Only map numeric stats for numeric variables
        self.generated_dictionary.loc[numeric_mask, 'Max'] = self.generated_dictionary.loc[numeric_mask, 'Variable'].map(max_values)
        self.generated_dictionary.loc[numeric_mask, 'Min'] = self.generated_dictionary.loc[numeric_mask, 'Variable'].map(min_values)
        self.generated_dictionary.loc[numeric_mask, 'Mean'] = self.generated_dictionary.loc[numeric_mask, 'Variable'].map(mean_values)
        self.generated_dictionary.loc[numeric_mask, 'Std'] = self.generated_dictionary.loc[numeric_mask, 'Variable'].map(std_values)
        self.generated_dictionary.loc[numeric_mask, '25%'] = self.generated_dictionary.loc[numeric_mask, 'Variable'].map(percentiles_25)
        self.generated_dictionary.loc[numeric_mask, '50%'] = self.generated_dictionary.loc[numeric_mask, 'Variable'].map(percentiles_50)
        self.generated_dictionary.loc[numeric_mask, '75%'] = self.generated_dictionary.loc[numeric_mask, 'Variable'].map(percentiles_75)

        # Add unique values for categorical types
        self.generated_dictionary['Unique Values'] = np.nan
        cat_mask = self.generated_dictionary['Type'].isin(['Discrete', 'Nominal'])

        for var in self.generated_dictionary.loc[cat_mask, 'Variable']:
            unique_vals = input_df[var].dropna().unique()
            if len(unique_vals) == 0:
                displayed = 'All NaN'
            else:
                unique_vals = sorted(unique_vals, key=lambda x: str(x))
                if len(unique_vals) > 20:
                    displayed = ', '.join(map(str, unique_vals[:20])) + ', ...'
                else:
                    displayed = ', '.join(map(str, unique_vals))
                    
            self.generated_dictionary.loc[
                self.generated_dictionary['Variable'] == var, 'Unique Values'
            ] = displayed

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

    def save_generated_dictionary(self, output_path = None, sheet_name='Data Dictionary'):
        """
        Save the generated data dictionary to an Excel file with formatted headers
        
        Parameters:
        -----------
        output_path : str
            Path to save the Excel file
        sheet_name : str, default 'Data Dictionary'
            Name of the worksheet
        """
        if self.generated_dictionary is None:
            raise ValueError("No generated dictionary available. Run generate_dataframe() first.")
        
        if output_path is None:
            output_path = self.default_dictionary_location + "/" +"data_dictionary.xlsx"

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Convert DataFrame to Excel
                self.generated_dictionary.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    startrow=1,
                    header=False
                )
                
                # Get workbook objects
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]
                
                # Define header format
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#4472C4',  # Blue background
                    'font_color': 'white',   # White text
                    'border': 1
                })
                
                # Write column headers with format
                for col_num, value in enumerate(self.generated_dictionary.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Add autofilter
                worksheet.autofilter(0, 0, 0, len(self.generated_dictionary.columns)-1)
                
                # Set column widths
                for idx, col in enumerate(self.generated_dictionary.columns):
                    max_len = max((
                        self.generated_dictionary[col].astype(str).map(len).max(),
                        len(col)
                    )) + 2
                    worksheet.set_column(idx, idx, max_len)
                    
                print(f"Data dictionary saved successfully to {output_path}")
                
        except Exception as e:
            print(f"Error saving data dictionary: {e}")
            raise