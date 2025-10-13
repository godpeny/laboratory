#!/usr/bin/env python3
"""
Simple and robust script to read a parquet file, duplicate the first row 5 times,
and save as a new parquet file.
"""

import pandas as pd
import os
import sys
from pathlib import Path


def duplicate_first_row_parquet(input_file: str, output_file: str, num_duplicates: int = 5, 
                               fill_null_values: bool = True, default_score: float = 0.1, 
                               default_tolerance: float = 0.1):
    """
    Read a parquet file, duplicate the first row specified number of times,
    and save as a new parquet file.
    
    Args:
        input_file (str): Path to input parquet file
        output_file (str): Path to output parquet file
        num_duplicates (int): Number of times to duplicate the first row (default: 5)
        fill_null_values (bool): Whether to fill null values in float columns (default: True)
        default_score (float): Default value for null score values (default: 0.1)
        default_tolerance (float): Default value for null toleranceScore values (default: 0.1)
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"Reading parquet file: {input_file}")
        # Read the parquet file
        df = pd.read_parquet(input_file)
        
        if df.empty:
            raise ValueError("Input parquet file is empty")
        
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle null values in the entire dataset if requested
        if fill_null_values:
            print("Handling null values in float columns for entire dataset...")
            original_nulls = df.isnull().sum()
            
            # Fill null values in score column
            if 'score' in df.columns and df['score'].isna().any():
                df['score'] = df['score'].fillna(default_score)
                print(f"  - Filled {original_nulls['score']} null 'score' values with: {default_score}")
            
            # Fill null values in toleranceScore column
            if 'toleranceScore' in df.columns and df['toleranceScore'].isna().any():
                df['toleranceScore'] = df['toleranceScore'].fillna(default_tolerance)
                print(f"  - Filled {original_nulls['toleranceScore']} null 'toleranceScore' values with: {default_tolerance}")
        
        # Get the first row (now with null values handled)
        first_row = df.iloc[0:1].copy()  # Keep as DataFrame to preserve structure
        
        print(f"First row to duplicate (after null handling):")
        print(first_row)
        
        # Create duplicates of the first row
        duplicated_rows = pd.concat([first_row] * num_duplicates, ignore_index=True)
        
        # Combine duplicated rows with the original dataframe
        result_df = pd.concat([duplicated_rows, df], ignore_index=True)
        
        print(f"New data shape after adding {num_duplicates} duplicates: {result_df.shape}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the result as a new parquet file with compatible format
        print(f"Saving to: {output_file}")
        
        # Use pyarrow to save with same format as original
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Read original file to get exact schema
        original_table = pq.read_table(input_file)
        original_schema = original_table.schema
        
        # Convert to pyarrow table
        table = pa.Table.from_pandas(result_df, preserve_index=False)
        
        # Create schema that matches original exactly (not null + field metadata)
        fields = []
        for i, field in enumerate(original_schema):
            # Use original field properties but update data from our table
            new_field = pa.field(field.name, field.type, nullable=field.nullable, metadata=field.metadata)
            fields.append(new_field)
        
        # Create new schema with exact same properties as original
        target_schema = pa.schema(fields)
        
        # Cast table to match original schema exactly
        table_with_original_schema = table.cast(target_schema)
        
        # Write with same settings as original
        pq.write_table(table_with_original_schema, output_file, 
                      write_statistics=True,
                      use_dictionary=True,
                      compression='snappy')
        
        print("✅ Successfully created new parquet file with duplicated first row!")
        return result_df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def main():
    """Main function to execute the script."""
    # Define file paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "result_list copy.parquet"
    output_file = script_dir / "result_list_with_duplicates.parquet"
    
    print("🚀 Starting parquet file processing...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    # Process the file
    result_df = duplicate_first_row_parquet(
        input_file=str(input_file),
        output_file=str(output_file),
        num_duplicates=5,
        fill_null_values=True,
        default_score=0.1,
        default_tolerance=0.1
    )
    
    print("-" * 50)
    print("📊 Summary:")
    print(f"   Original rows: {len(result_df) - 5}")
    print(f"   Duplicated rows: 5")
    print(f"   Total rows: {len(result_df)}")
    print(f"   Output saved to: {output_file}")


if __name__ == "__main__":
    main()
