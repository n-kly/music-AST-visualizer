import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Define the paths to the data directory and the output directory for MP3 files
data_directory = '../../../music_genres/data/'
output_directory = './songs1'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through all parquet files in the data directory
for file_name in tqdm(os.listdir(data_directory), desc="Processing Parquet Files"):
    if file_name.endswith('.parquet'):
        parquet_file_path = os.path.join(data_directory, file_name)
        
        # Read the parquet file
        table = pq.read_table(parquet_file_path)
        df = table.to_pandas()

        # Assume the dataframe has a column 'audio' with binary data of MP3 files
        # and a column 'song_id' with the desired file names for the MP3 files
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {file_name}"):
            mp3_data = row['audio']['bytes']
            mp3_file_name = row['song_id']
            
            # Save the MP3 file
            output_file_path = os.path.join(output_directory, str(mp3_file_name) + ".mp3")
            with open(output_file_path, 'wb') as f:
                f.write(mp3_data)

print(f"Extracted MP3 files to {output_directory}")