import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

class Report:
    """
    Class for generating various analytical reports
    """
    def __init__(self):
        self.default_formats = {
            'header': {
                'bold': True,
                'bg_color': '#0066cc',
                'font_color': 'white'
            },
            'number': {'num_format': '0.00'},
            'percent': {'num_format': '0.00%'}
        }

    def export_csi_analysis_to_excel(self,
                                   csi_results: Dict,
                                   summary: pd.DataFrame,
                                   output_path: Optional[str] = None,
                                   baseline_name: str = "Baseline",
                                   validation_name: str = "Validation") -> str:
        """
        Export CSI analysis results to Excel with multiple sheets
        
        Parameters:
        -----------
        csi_results : dict
            Results from calculate_csi_v2
        summary : pd.DataFrame
            Summary DataFrame from analyze_csi_results
        output_path : str, optional
            Path to save Excel file
        baseline_name : str
            Name of baseline sample for reporting
        validation_name : str
            Name of validation sample for reporting
            
        Returns:
        --------
        str
            Path to the generated Excel file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'csi_analysis_{timestamp}.xlsx'
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Get workbook and create formats
            workbook = writer.book
            formats = self._create_excel_formats(workbook)
            
            # Create sheets
            self._create_summary_sheet(summary, writer, formats)
            self._create_detailed_sheet(csi_results, writer, formats, 
                                      baseline_name, validation_name)
            self._create_feature_sheets(csi_results, writer, formats)
            
        return output_path

    def _create_excel_formats(self, workbook) -> Dict:
        """Create Excel formats for styling"""
        return {
            'header': workbook.add_format(self.default_formats['header']),
            'number': workbook.add_format(self.default_formats['number']),
            'percent': workbook.add_format(self.default_formats['percent'])
        }

    def _create_summary_sheet(self, 
                            summary: pd.DataFrame,
                            writer: pd.ExcelWriter,
                            formats: Dict):
        """Create summary sheet"""
        summary.to_excel(writer, sheet_name='Summary', index=False)
        sheet = writer.sheets['Summary']
        
        # Format headers and columns
        for col_num, value in enumerate(summary.columns.values):
            sheet.write(0, col_num, value, formats['header'])
            sheet.set_column(col_num, col_num, 15)

    def _create_detailed_sheet(self,
                             csi_results: Dict,
                             writer: pd.ExcelWriter,
                             formats: Dict,
                             baseline_name: str,
                             validation_name: str):
        """Create detailed analysis sheet"""
        detailed_rows = []
        for feature, result in csi_results.items():
            if result['csi'] is not None:
                details = result['details']
                top_contributors = details.nlargest(3, 'Contribution_Pct')
                
                detailed_rows.append({
                    'Feature': feature,
                    'Type': 'Categorical' if result['is_categorical'] else 'Numeric',
                    'CSI Value': result['csi'],
                    'Status': result['status'],
                    'Top Contributing Category/Bin': str(top_contributors.index[0]),
                    'Top Contribution %': top_contributors.iloc[0]['Contribution_Pct'] / 100,
                    f'{baseline_name} %': top_contributors.iloc[0]['Baseline_Dist'],
                    f'{validation_name} %': top_contributors.iloc[0]['Validation_Dist']
                })
        
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
        sheet = writer.sheets['Detailed_Analysis']
        
        # Format headers and columns
        for col_num, value in enumerate(detailed_df.columns.values):
            sheet.write(0, col_num, value, formats['header'])
            sheet.set_column(col_num, col_num, 20)
        
        # Apply number formats
        sheet.set_column('C:C', None, formats['number'])  # CSI Value
        sheet.set_column('F:H', None, formats['percent'])  # Percentages

    def _create_feature_sheets(self,
                             csi_results: Dict,
                             writer: pd.ExcelWriter,
                             formats: Dict):
        """Create individual feature sheets"""
        for feature, result in csi_results.items():
            if result['csi'] is not None:
                sheet_name = f'{feature[:28]}_Analysis'
                details_df = result['details'].reset_index()
                details_df.columns = ['Bin/Category' if col == 'index' else col 
                                    for col in details_df.columns]
                
                # Add feature metadata
                metadata_df = pd.DataFrame([
                    ['Feature', feature],
                    ['Type', 'Categorical' if result['is_categorical'] else 'Numeric'],
                    ['CSI Value', result['csi']],
                    ['Status', result['status']],
                    ['', ''],  # Empty row for spacing
                ], columns=['Metric', 'Value'])
                
                # Write metadata
                metadata_df.to_excel(writer, sheet_name=sheet_name, index=False)
                sheet = writer.sheets[sheet_name]
                
                # Write details table
                details_df.to_excel(writer, sheet_name=sheet_name, 
                                  startrow=len(metadata_df) + 2, index=False)
                
                # Format sheet
                sheet.set_column('A:A', 20)
                sheet.set_column('B:Z', 15)
                
                # Format headers
                for col_num, value in enumerate(details_df.columns.values):
                    sheet.write(len(metadata_df) + 2, col_num, value, formats['header'])
                
                # Apply number formats to details table
                detail_start_row = len(metadata_df) + 3
                sheet.set_column('B:C', None, formats['percent'])  # Distribution columns
                sheet.set_column('D:D', None, formats['number'])   # Difference
                sheet.set_column('E:F', None, formats['number'])   # Log ratio and Component
                sheet.set_column('G:G', None, formats['percent'])  # Contribution %

    def generate_csi_report(self,
                          baseline_sample: pd.DataFrame,
                          validation_sample: pd.DataFrame,
                          features: list,
                          categorical_features: Optional[list] = None,
                          output_path: Optional[str] = None,
                          baseline_name: str = "Baseline",
                          validation_name: str = "Validation") -> Tuple[Dict, pd.DataFrame, str]:
        """
        Generate comprehensive CSI analysis report
        
        Parameters:
        -----------
        baseline_sample : pd.DataFrame
            Baseline dataset
        validation_sample : pd.DataFrame
            Validation dataset
        features : list
            List of features to analyze
        categorical_features : list, optional
            List of categorical features
        output_path : str, optional
            Path to save Excel report
        baseline_name : str
            Name of baseline sample
        validation_name : str
            Name of validation sample
            
        Returns:
        --------
        Tuple containing:
            - CSI results dictionary
            - Summary DataFrame
            - Path to Excel report
        """
        from .performance import calculate_csi_v2, analyze_csi_results
        
        # Calculate CSI
        csi_results = calculate_csi_v2(
            baseline_sample=baseline_sample,
            validation_sample=validation_sample,
            features=features,
            categorical_features=categorical_features,
            bins=10
        )
        
        # Analyze results
        summary = analyze_csi_results(
            csi_results=csi_results,
            output_path='outputs/csi_analysis' if output_path else None,
            plot=True
        )
        
        # Export to Excel
        excel_path = self.export_csi_analysis_to_excel(
            csi_results=csi_results,
            summary=summary,
            output_path=output_path,
            baseline_name=baseline_name,
            validation_name=validation_name
        )
        
        return csi_results, summary, excel_path 