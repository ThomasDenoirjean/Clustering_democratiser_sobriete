import pandas as pd
import re
import json

def clean_and_parse_json(json_string):
    try:
        # Skip invalid strings
        if not json_string.strip().startswith("{"):
            return None
        # Remove trailing commas
        cleaned_string = re.sub(r",\s*}", "}", json_string)
        cleaned_string = re.sub(r",\s*]", "]", cleaned_string)
        # Parse the cleaned JSON string
        return json.loads(cleaned_string)
    except json.JSONDecodeError:
        return None

def is_meaningful_json(parsed_data):
    if not isinstance(parsed_data, dict):
        return False
    # Check if the JSON only contains "None" values
    for key, value in parsed_data.items():
        if key != "None" or (isinstance(value, dict) and any(k != "None" for k in value.keys())):
            return True
    return False

def extract_items(dataframe):
    new_rows = []

    for idx, row in dataframe.iterrows():
        extracted_data = row['FACTOR']

        parsed_data = clean_and_parse_json(extracted_data)

        if parsed_data:
            for factor in list(parsed_data.keys()):
                new_rows.append({
                    'index': idx,  # Use original index as reference
                    'ITEM': row['ITEM'],
                    'FACTOR': factor,
                    'doi': row['doi']
                })
        elif row['FACTOR'] != '\"None\"':
            print(row['FACTOR'])

    # Create new DataFrame
    new_df = pd.DataFrame(new_rows)

    return new_df
