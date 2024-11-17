import pandas as pd
from typing import Union, List, Any
from .filters import Conditions, Demographics, Offenses
from datetime import datetime, timedelta
import os
import traceback
import copy

###################################################################################

# SAMPLE OUTPUT (Pydantic)
# eligibility_filters = [
#  Conditions(
#         demographics=Demographics(
#         individual_age= 0,
#         individual_age_column= "",
#         individual_age_operator= "==",
#         individual_location= "",
#         individual_location_column= "current location",
#         sentenced_by= "Los Angeles",
#         sentenced_by_column= "controlling case sentencing county",
#         individual_ethnicity= "Hispanic",
#         individual_ethnicity_column= "ethnicity",
#         sentenced_age= 0,
#         sentenced_age_column= "",
#         sentenced_age_operator= "",
#         sentenced_years= 20,
#         sentenced_years_column= "aggregate sentence in years",
#         sentenced_years_operator= ">",
#         served_years= 10,
#         served_years_column= "time served in years",
#         served_years_operator= ">=",
#         demographic_subquery= "Find all individuals who are Hispanic, sentenced to over 20 years, served at least 10 years and are from Los Angeles County."
#         ),
        
#         offenses=Offenses( 
#         offense_number= "PC666",
#         offense_number_column= "",
#         offense_number_tables= "Table A, B",
#         offense_number_type= "current",
#         offense_number_operator= "include",
#         offense_description= "",
#         offense_description_column= "controlling offense cleaned",
#         offense_subquery= "I would also like to see those who have a current offense in Table A, B of the selection criteria"
#         ),
        
#         logical_operations= ["AND","AND","AND","AND"]
#     )
#   ]

# df = pd.read_excel(r"C:\Users\aparn\llm_code_dev\data\Demographics.xlsx")

class CohortProcessor:
    def __init__(self, df, eligibility_filters) -> None:
        print(f"eligibility_filters  type --> {type(eligibility_filters)}")
        print(f"eligibility_filters --> {eligibility_filters}")
        
        self.df = df
        self.cohort = copy.deepcopy(self.df)
        self.filters = eligibility_filters['eligibility_filters']
        self.demographics_filters = self.filters[0]['demographics']
        self.offenses_filters = self.filters[0]['offenses']

    def get_var_groups(self, category):
        if category == 'demographics':
            filters = self.demographics_filters
        elif category == 'offenses':
            filters = self.offenses_filters
        else: 
            return
        
        # Find all the variables to query df on
        typ = []
        for v in filters.keys():
            vs = v.split('_')
            if (len(vs) == 2) and ("subquery" not in vs): 
                typ.append(v)
        # Group the column, value and operator for each variable
        gp = []
        for v in typ:
            vgp = [v]
            if v+"_operator" in filters.keys():
                vgp.append(v+"_operator")
            if v+"_column" in filters.keys():
                vgp.append(v+"_column")
            if v+"_type" in filters.keys():
                vgp.append(v+"_type")
            if v+"_tables" in filters.keys():
                vgp.append(v+"_tables")
            gp.append(vgp)
        
        return gp
    
    def apply_demographics_filters(self, var_groups):
        for vgp in var_groups: 
            if all(self.demographics_filters[v] != "" for v in vgp):
                v = [v for v in vgp if ("_operator" not in v) and ("_column" not in v)][0]
                if (v+"_operator" not in vgp) or (self.demographics_filters[v+"_operator"] == 'exact') or (self.demographics_filters[v+"_operator"] == '=='):
                    self.cohort = self.cohort[self.cohort[self.demographics_filters[v+"_column"]] == self.demographics_filters[v]]
                elif self.demographics_filters[v+"_operator"] == '>=':
                    self.cohort = self.cohort[self.cohort[self.demographics_filters[v+"_column"]] >= self.demographics_filters[v]]
                elif self.demographics_filters[v+"_operator"] == '>':
                    self.cohort = self.cohort[self.cohort[self.demographics_filters[v+"_column"]] > self.demographics_filters[v]]       
                elif self.demographics_filters[v+"_operator"] == '<=':
                    self.cohort = self.cohort[self.cohort[self.demographics_filters[v+"_column"]] <= self.demographics_filters[v]]        
                elif self.demographics_filters[v+"_operator"] == '<':
                    self.cohort = self.cohort[self.cohort[self.demographics_filters[v+"_column"]] < self.demographics_filters[v]]        
                elif self.demographics_filters[v+"_operator"] == '!=':
                    self.cohort = self.cohort[self.cohort[self.demographics_filters[v+"_column"]] != self.demographics_filters[v]] 
                print(v, len(self.cohort))
        return self.cohort
    
    def apply_offenses_filters(self, var_groups):
        for vgp in var_groups: 
            if all(self.offenses_filters[v] != "" for v in vgp):
                v = [v for v in vgp if ("_operator" not in v) and ("_column" not in v)][0]
                if (v+"_operator" not in vgp) or (self.offenses_filters[v+"_operator"] == 'exact'):
                    self.cohort = self.cohort[self.cohort[self.offenses_filters[v+"_column"]] == self.offenses_filters[v]]
                elif self.offenses_filters[v+"_operator"] == 'includes':
                    self.cohort = self.cohort[self.cohort[self.offenses_filters[v+"_column"]].isin([])]
                elif self.offenses_filters[v+"_operator"] == 'excludes':
                    self.cohort = self.cohort[~self.cohort[self.offenses_filters[v+"_column"]].isin([])]
                print(v, len(self.cohort))
        return self.cohort
    
    def generate_cohort(self, category = None):
        if not category: 
           gp = self.get_var_groups(category = 'demographics') 
           _ = self.apply_demographics_filters(gp)
           gp = self.get_var_groups(category = 'offenses') 
           _ = self.apply_offenses_filters(gp)
        else: 
            gp = self.get_var_groups(category = category) 
            _ = self.apply_demographics_filters(gp)
        
        return self.cohort
    
###################################################################################

# class CohortProcessor:
#     def __init__(self, data_df, eligibility_filters) -> None:
#         self.df = data_df
#         self.filters = eligibility_filters


#     def detect_file_type(self) -> str:
#         """
#         Detect if the file is demographics or offenses based on columns present.
#         Returns: 'demographics' or 'offenses'
#         """
#         # Required columns for each type
#         demographics_columns = {'Ethnicity', 'Aggregate Sentence in Months'}
#         offenses_columns = {'Offense Category', 'Controlling Offense'}
        
#         # Check which columns are present
#         demographics_present = demographics_columns.intersection(set(self.df.columns))
#         offenses_present = offenses_columns.intersection(set(self.df.columns))
        
#         print(f"\nFile type detection:")
#         print(f"Demographics columns found: {demographics_present}")
#         print(f"Offenses columns found: {offenses_present}")
        
#         # If the file has Ethnicity and Aggregate Sentence in Months, it's definitely demographics
#         if 'Ethnicity' in self.df.columns and 'Aggregate Sentence in Months' in self.df.columns:
#             print("Detected as demographics file based on presence of Ethnicity and Aggregate Sentence columns")
#             return 'demographics'
#         # If it has Offense Category and Controlling Offense, it's offenses
#         elif 'Offense Category' in self.df.self.columns and 'Controlling Offense' in self.df.self.columns:
#             print("Detected as offenses file based on presence of Offense Category and Controlling Offense columns")
#             return 'offenses'
#         # Default to demographics if Ethnicity is present
#         elif 'Ethnicity' in self.df.self.columns:
#             print("Defaulting to demographics file based on presence of Ethnicity column")
#             return 'demographics'
#         else:
#             print("Defaulting to offenses file")
#             return 'offenses'


#     def check_column_exists(self, column: str) -> bool:
#         """Verify if a column exists in the dataframe."""
#         if column not in self.df.columns:
#             print(f"Warning: Column '{column}' not found in dataset. Skipping this filter.")
#             return False
#         return True


#     def apply_filter(self, column: str, value: Union[int, float, str, bool], operator: str) -> pd.Series:
#         """Apply filter with proper type handling."""
#         try:
#             column_data = self.df[column]
            
#             # Handle date comparisons
#             if column == 'Birthday':
#                 column_data = pd.to_datetime(column_data, errors='coerce')
#                 if operator == '>':
#                     cutoff_date = datetime.now() - timedelta(days=value*365.25)
#                     return column_data < cutoff_date  # Reversed because older date means higher age
#                 elif operator == '<':
#                     cutoff_date = datetime.now() - timedelta(days=value*365.25)
#                     return column_data > cutoff_date  # Reversed because older date means higher age
            
#             # Handle regular comparisons
#             if operator == '>':
#                 return column_data > value
#             elif operator == '<':
#                 return column_data < value
#             elif operator in ['=', '==']:
#                 return column_data == value
#             elif operator == '!=':
#                 return column_data != value
#             elif operator == 'contains':
#                 if isinstance(value, (list, str)):
#                     return column_data.astype(str).str.contains(str(value), na=False)
#                 return pd.Series([False] * len(self.df))
#             elif operator == 'not_contains':
#                 if isinstance(value, (list, str)):
#                     return ~column_data.astype(str).str.contains(str(value), na=False)
#                 return pd.Series([True] * len(self.df))
#             else:
#                 print(f"Unsupported operator '{operator}'")
#                 return pd.Series([False] * len(self.df))
                
#         except Exception as e:
#             print(f"Error in apply_filter for {column}: {str(e)}")
#             return pd.Series([False] * len(self.df))


#     def process_demographics(self, demographics: Demographics) -> pd.Series:
#         """Process demographics filters from eligibility_filters."""
#         valid_rows = pd.Series([True] * len(self.df), index=self.df.index)
        
#         # Get all attributes from the Demographics object
#         filter_attrs = [attr for attr in dir(demographics) if not attr.startswith('__') and not attr.endswith('_column') and not attr.endswith('_operator') and not attr == 'demographics_subquery']
        
#         for attr in filter_attrs:
#             value = getattr(demographics, attr)
#             column_attr = f"{attr}_column"
#             operator_attr = f"{attr}_operator"
            
#             # Get the column name
#             column = getattr(demographics, column_attr, None)
#             if not column:
#                 continue
                
#             # Get the operator (default to '==' if not specified)
#             operator = getattr(demographics, operator_attr, '==') if hasattr(demographics, operator_attr) else '=='
            
#             if self.check_column_exists(self.df, column):
#                 filter_result = self.apply_filter(column, value, operator)
#                 valid_rows &= filter_result
#                 print(f"Applied {column} filter: {operator} {value} - Matches: {filter_result.sum()}")
        
#         return valid_rows


#     def process_offenses(self, offenses: Offenses) -> pd.Series:
#         """Process offenses filters from eligibility_filters."""
#         valid_rows = pd.Series([True] * len(self.df), index=self.df.index)
        
#         if not self.check_column_exists('Offense Category'):
#             return valid_rows
        
#         # Get tables directly from the Offenses object
#         if offenses.include_tables:
#             include_tables = [table.strip() for table in offenses.include_tables.split(',')]
#             include_mask = self.df['Offense Category'].isin(include_tables)
#             valid_rows &= include_mask
#             print(f"Applied include tables filter ({offenses.include_tables}) - Matches: {include_mask.sum()}")
        
#         if offenses.exclude_tables:
#             exclude_tables = [table.strip() for table in offenses.exclude_tables.split(',')]
#             exclude_mask = ~self.df['Offense Category'].isin(exclude_tables)
#             valid_rows &= exclude_mask
#             print(f"Applied exclude tables filter ({offenses.exclude_tables}) - Matches: {exclude_mask.sum()}")
        
#         return valid_rows


#     def process_offenses(self, offenses: Offenses) -> pd.Series:
#         """Process offenses filters separately."""
#         valid_rows = pd.Series([True] * len(self.df), index=self.df.index)
        
#         if not self.check_column_exists('Offense Category'):
#             return valid_rows
            
#         # Process include tables
#         if offenses.include_tables:
#             include_tables = [table.strip() for table in offenses.include_tables.split(',')]
#             include_mask = self.df['Offense Category'].isin(include_tables)
#             valid_rows &= include_mask
#             print(f"Applied include tables filter - Matches: {include_mask.sum()}")
        
#         # Process exclude tables
#         if offenses.exclude_tables:
#             exclude_tables = [table.strip() for table in offenses.exclude_tables.split(',')]
#             exclude_mask = ~self.df['Offense Category'].isin(exclude_tables)
#             valid_rows &= exclude_mask
#             print(f"Applied exclude tables filter - Matches: {exclude_mask.sum()}")
        
#         return valid_rows


#     def process_generic_filters(self, obj: Any, unique_id_column: str) -> pd.DataFrame:
#         """Process filters with proper error handling and logging."""
#         print(f"\nProcessing filters for {obj.__class__.__name__}")
#         valid_rows = pd.Series([True] * len(self.df), index=self.df.index)
        
#         # Detect file type
#         file_type = self.detect_file_type(self.df)
#         print(f"Detected file type: {file_type}")
        
#         if file_type == 'demographics':
#             if hasattr(obj, 'demographics') and obj.demographics:
#                 print("\nApplying Demographics filters:")
#                 demographics_mask = self.process_demographics(obj.demographics, self.df)
#                 valid_rows &= demographics_mask
#                 print(f"Rows after demographics filters: {valid_rows.sum()}/{len(self.df)}")
#             else:
#                 print("No demographics filters to apply")
#                 return self.df
#         else:  # offenses
#             if hasattr(obj, 'offenses') and obj.offenses:
#                 print("\nApplying Offenses filters:")
#                 offenses_mask = self.process_offenses(obj.offenses)
#                 valid_rows &= offenses_mask
#                 print(f"Rows after offenses filters: {valid_rows.sum()}/{len(self.df)}")
#             else:
#                 print("No offenses filters to apply")
#                 return self.df
        
#         return self.df[valid_rows]


#     def process_eligibility_filters(self, unique_id_column: str) -> pd.DataFrame:
#         """Apply all eligibility filters with proper logging."""
#         if not self.filters:
#             print("Warning: No filters provided")
#             return self.df
            
#         print(f"\nProcessing {len(self.filters)} filter conditions")
#         result_df = self.df.copy()
        
#         for i, condition in enumerate(self.filters, 1):
#             print(f"\nProcessing condition {i}/{len(self.filters)}")
#             result_df = self.process_generic_filters(condition, result_df, unique_id_column)
            
#         return result_df

#     def run_cohort_processor(self):
#         try:
#             print("\nDataset Information:")
#             print(f"Original shape: {self.df.shape}")
#             print(f"Columns: {', '.join(self.df.columns.tolist())}")
            
#             # Print sample of data for verification
#             print("\nFirst few rows of key columns:")
#             display_columns = ['Ethnicity', 'Offense Category', 'Aggregate Sentence in Months']
#             if all(col in self.df.columns for col in display_columns):
#                 print(self.df[display_columns].head().to_string())
            
#             # Detect file type before processing
#             file_type = self.detect_file_type(self.df)
#             print(f"\nProcessing file as: {file_type}")
            
#             filtered_df = self.process_eligibility_filters(eligibility_filters, self.df, 'CDCNo')
            
#             print("\nFiltering Results:")
#             print(f"Original records: {len(self.df)}")
#             print(f"Filtered records: {len(filtered_df)}")
#             print(f"Reduction: {((len(self.df) - len(filtered_df))/len(self.df))*100:.2f}%")
            
#             return filtered_df
        
#         except Exception as e:
#             print(f"\nError in run_cohort_processor: {str(e)}")
#             print(traceback.format_exc())


def do_dry_run(self):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(script_dir, '../prior_commitments.xlsx')
        
    print("\n=== Starting Eligibility Processing ===")
    print(f"Reading data from: {excel_file}")
    df = pd.read_excel(excel_file)

    cohort_processor = CohortProcessor(data_df=df, eligibility_filters=eligibility_filters)
    output_df = cohort_processor.run_cohort_processor()

    output_file = os.path.join(script_dir, 'filtered_results.xlsx')
    output_df.to_excel(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
        

if __name__ == "__main__":
    do_dry_run()