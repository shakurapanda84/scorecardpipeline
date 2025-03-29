import pandas as pd
from IPython.display import HTML, display
import numpy as np

def get_target(df):
    """
    Create target variable based on loan status
    """
    df['target'] = df['loan_status'].map({'Fully Paid':0, 'Current':0, 'In Grace Period':0,
                                         'Late (16-30 days)':1,'Late (31-120 days)':1,'Default':1,
                                         'Charged Off':1})
    
    df.drop('loan_status',axis=1,inplace=True)
    return df

def display_df(df, 
               output_format='table',
               headers_style=None,
               precision=2,
               width=None,
               max_rows=None,
               max_cols=None,
               index=True,
               gradient_cols=None,
               highlight_max=None,
               highlight_min=None,
               bar_cols=None,
               custom_styles=None):
    """
    Customizable function to display DataFrames in Jupyter notebooks.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to display
    output_format : str, default 'table'
        Options: 'table', 'html', 'latex', 'markdown'
    headers_style : dict, default None
        CSS styles for headers (e.g., {'color': 'white', 'background-color': '#000066'})
    precision : int, default 2
        Number of decimal places for floating point numbers
    width : int, default None
        Max width of the display
    max_rows : int, default None
        Maximum number of rows to display
    max_cols : int, default None
        Maximum number of columns to display
    index : bool, default True
        Whether to show index
    gradient_cols : list, default None
        Columns to apply color gradient
    highlight_max : list, default None
        Columns where maximum values should be highlighted
    highlight_min : list, default None
        Columns where minimum values should be highlighted
    bar_cols : list, default None
        Columns to display as bars
    custom_styles : dict, default None
        Additional custom styles to apply
    
    Returns:
    --------
    Styled DataFrame display in specified format
    """
    
    # Create a copy to avoid modifying original
    df_display = df.copy()
    
    # Set display options
    pd.set_option('display.precision', precision)
    if width:
        pd.set_option('display.width', width)
    if max_rows:
        pd.set_option('display.max_rows', max_rows)
    if max_cols:
        pd.set_option('display.max_columns', max_cols)
    
    # Initialize styler
    styler = df_display.style
    
    # Format floating point numbers
    styler.format(precision=precision)
    
    # Apply header styles
    if headers_style:
        header_styles = [headers_style for _ in range(len(df_display.columns))]
        styler.set_table_styles([
            {'selector': 'th',
             'props': [(k, v) for k, v in headers_style.items()]}
        ])
    
    # Apply gradient colors
    if gradient_cols:
        for col in gradient_cols:
            if col in df_display.columns:
                styler.background_gradient(cmap='YlOrRd', subset=[col])
    
    # Highlight maximum values
    if highlight_max:
        styler.highlight_max(subset=highlight_max, color='lightgreen')
    
    # Highlight minimum values
    if highlight_min:
        styler.highlight_min(subset=highlight_min, color='lightpink')
    
    # Add bars
    if bar_cols:
        styler.bar(subset=bar_cols, color=['#d65f5f', '#5fba7d'])
    
    # Apply custom styles
    if custom_styles:
        for selector, props in custom_styles.items():
            styler.set_table_styles([{'selector': selector, 'props': props}])
    
    # Output in specified format
    if output_format.lower() == 'html':
        return HTML(styler.to_html())
    elif output_format.lower() == 'latex':
        return styler.to_latex()
    elif output_format.lower() == 'markdown':
        return styler.to_markdown()
    else:  # default table format
        return display(styler)

# Example usage:
if __name__ == "__main__":
    # Create sample DataFrame
    df = pd.DataFrame({
        'A': np.random.randn(10),
        'B': np.random.randn(10),
        'C': np.random.randn(10),
        'D': np.random.choice(['X', 'Y', 'Z'], 10)
    })
    
    # Example 1: Basic table with styled headers
    display_df(df, 
              headers_style={'color': 'white', 
                           'background-color': '#000066'})
    
    # Example 2: With gradients and highlights
    display_df(df,
              gradient_cols=['A'],
              highlight_max=['B'],
              highlight_min=['C'],
              headers_style={'color': 'white', 
                           'background-color': '#006600'})
    
    # Example 3: With bars and custom styling
    custom_styles = {
        'td': [('background-color', '#f0f0f0'),
               ('color', '#333333')],
        'tr:hover': [('background-color', '#ffffb3')]
    }
    
    display_df(df,
              bar_cols=['A', 'B'],
              custom_styles=custom_styles,
              headers_style={'color': 'white', 
                           'background-color': '#660066'})
    



def get_iv_woe(data, target, bins=10, show_woe=False):
    """
    Calculate information value and WOE for features
    """
    
    # Empty dataframes
    iv_df = pd.DataFrame()
    woe_dct = {}
    
    # Extract column names
    cols = data.columns
    
    # Run WOE and IV on all columns
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            
        else:
            binned_x = data[ivars]
        
        # Calculate WOE and IV for categorical columns
        if data[ivars].dtype.name == 'object' or data[ivars].dtype.name == 'category':
            
            # Calculate counts for target=0 and target=1
            d0 = pd.DataFrame({'x': binned_x, 'y': target}).groupby('x')['y'].agg(['count', lambda x: (x==0).sum()]).reset_index()
            d0.columns = ['Cutoff', 'N', 'N0']
            d0['N1'] = d0['N'] - d0['N0']
            
            # Calculate proportions
            d0['Prop0'] = d0['N0']/d0['N0'].sum()
            d0['Prop1'] = d0['N1']/d0['N1'].sum()
            
            # Calculate WOE and add it to the dataframe
            d0['WoE'] = np.log(d0['Prop0']/d0['Prop1'])
            d0['IV'] = (d0['Prop0']-d0['Prop1'])*np.log(d0['Prop0']/d0['Prop1'])
            
            # Store the WoE values in a dictionary
            woe_dct[ivars] = dict(zip(d0['Cutoff'], d0['WoE']))
            
            # Show WOE table if requested
            if show_woe:
                print(ivars)
                print(d0)
                print("\n")
            
            # Store the IV value
            iv = d0['IV'].sum()
            iv_df = pd.concat([iv_df,pd.DataFrame({'Variable': [ivars], 'IV': [iv]})])
            
        # Calculate WOE and IV for numeric columns
        else:
            
            # Calculate counts for target=0 and target=1
            d0 = pd.DataFrame({'x': binned_x, 'y': target}).groupby('x')['y'].agg(['count', lambda x: (x==0).sum()]).reset_index()
            d0.columns = ['Cutoff', 'N', 'N0']
            d0['N1'] = d0['N'] - d0['N0']
            
            # Calculate proportions
            d0['Prop0'] = d0['N0']/d0['N0'].sum()
            d0['Prop1'] = d0['N1']/d0['N1'].sum()
            
            # Calculate WOE and add it to the dataframe
            d0['WoE'] = np.log(d0['Prop0']/d0['Prop1'])
            d0['IV'] = (d0['Prop0']-d0['Prop1'])*np.log(d0['Prop0']/d0['Prop1'])
            
            # Store the WoE values in a dictionary
            woe_dct[ivars] = dict(zip(d0['Cutoff'], d0['WoE']))
            
            # Show WOE table if requested
            if show_woe:
                print(ivars)
                print(d0)
                print("\n")
            
            # Store the IV value
            iv = d0['IV'].sum()
            iv_df = pd.concat([iv_df,pd.DataFrame({'Variable': [ivars], 'IV': [iv]})])
            
    # Sort IV values in descending order
    iv_df = iv_df.sort_values('IV', ascending=False).reset_index(drop=True)
    return iv_df, woe_dct

def replace_woe(data, woe_dct):
    """
    Replace values with WOE
    """
    cols = data.columns
    for ivars in cols:
        if ivars in woe_dct.keys():
            woe_dict = woe_dct[ivars]
            data[ivars] = data[ivars].map(woe_dict)
    return data
