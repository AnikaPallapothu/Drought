import os
import pandas as pd

def read_csv_with_non_empty_rows_and_progress(filename):
    filtered_df = pd.DataFrame()
    chunksize = 100000
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        filtered_chunk = chunk.dropna(how='any')
        print(f"Processed {len(filtered_chunk)} rows so far")
        filtered_df = pd.concat([filtered_df, filtered_chunk], ignore_index=True)
        if filtered_df.shape[0] > 100000:
            break
    return filtered_df

def prepare_data():

    soil_df = pd.read_csv("../data/raw/soil_data.csv")
    drought_df = read_csv_with_non_empty_rows_and_progress("../data/raw/train_timeseries.csv")
    
    drought_df['year'] = pd.DatetimeIndex(drought_df['date']).year
    drought_df['month'] = pd.DatetimeIndex(drought_df['date']).month
    drought_df['day'] = pd.DatetimeIndex(drought_df['date']).day
    drought_df['score'] = drought_df['score'].round().astype(int)
    
    merged_df = drought_df.merge(soil_df, on='fips')
    merged_df.to_csv('../data/processed/drought_data.csv', index=False)

if __name__ == "__main__":
    prepare_data()
