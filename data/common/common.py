import pandas as pd
import numpy as np
from IPython.display import HTML, display

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
               custom_styles=None,
               percentage_cols=None,
               percentage_precision=2,
               column_precisions=None):
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
    percentage_cols : list, default None
        Columns to format as percentages
    percentage_precision : int, default 2
        Number of decimal places for percentage columns
    column_precisions : dict, default None
        Dictionary specifying precision for individual columns
    
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

    # Hide index if specified
    if not index:
        styler = styler.hide(axis='index')
    
    # Format percentage columns
    if percentage_cols:
        for col in percentage_cols:
            if col in df_display.columns:
                styler.format({col: f'{{:.{percentage_precision}f}}%'})
    
    # Apply individual column precisions
    if column_precisions:
        for col, col_precision in column_precisions.items():
            if col in df_display.columns:
                styler.format({col: f'{{:.{col_precision}f}}'})
    
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

def init_notebook():
    """
    Initialize Jupyter notebook settings for DataFrame display and custom styles.
    """
    # Set pandas display options
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.width', 1000)  # Set display width
    pd.set_option('display.precision', 2)  # Set precision for floating point numbers

    # Define default custom styles
    default_custom_styles = {
        'td': [('background-color', '#f0f0f0'), ('color', '#333333')],
        'tr:hover': [('background-color', '#ffffb3')],
        'th': [('color', 'white'), ('background-color', '#000066')]
    }

    # Return the default custom styles for use in display_df
    return default_custom_styles
