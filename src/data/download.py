import os
import json
import subprocess
from tqdm import tqdm

# Function to read popular songs from a text file
def read_popular_songs(file_path):
    with open(file_path, 'r') as file:
        songs = file.readlines()
    return [song.strip() for song in songs]

# Download preview using ytmdl and convert to 16kHz mono MP3
def download_and_convert(song_name, output_dir='previews'):
    temp_output_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Download the song using ytmdl to the temporary directory
    command_download = f'ytmdl "{song_name}" -q -o {temp_output_dir}'
    subprocess.run(command_download, shell=True, check=True)
    
    # Find the downloaded file (assuming there's only one file in the temp directory)
    downloaded_files = os.listdir(temp_output_dir)
    if not downloaded_files:
        raise FileNotFoundError(f"No files found in {temp_output_dir} after download.")
    
    temp_output = os.path.join(temp_output_dir, downloaded_files[0])
    final_output = os.path.join(output_dir, f"{song_name.replace(' ', '_')}.mp3")

    # Convert the audio to 16kHz mono MP3 using ffmpeg
    command_convert = f'ffmpeg -i "{temp_output}" -ac 1 -ar 16000 "{final_output}"'
    subprocess.run(command_convert, shell=True, check=True)

    # Remove the temporary files
    for file in downloaded_files:
        os.remove(os.path.join(temp_output_dir, file))

# Main function
def main():
    # File containing the list of popular songs in the pop genre
    popular_songs_file = './popular_songs.txt'
    
    # Read popular songs from the text file
    popular_songs = read_popular_songs(popular_songs_file)
    
    all_songs = []
    artists_metadata = {}
    genres_metadata = {}
    
    # Create directories if they don't exist
    os.makedirs('previews', exist_ok=True)
    os.makedirs('metadata', exist_ok=True)
    
    for song_name in tqdm(popular_songs, desc="Downloading popular songs"):
        # Download and convert preview
        download_and_convert(song_name)
        
        # For simplicity, we'll assume the song metadata is available after downloading
        # In practice, you might need to parse the downloaded files or use another method to get metadata
        song_metadata = {
            'name': song_name,
            'genres': 'pop'  # Assigning the genre as 'pop' since all songs are from the pop genre
            # Add other metadata fields as needed
        }
        all_songs.append(song_metadata)
        
        # Add to artists metadata
        artist_name = song_name.split('-')[0].strip()  # This is a simplistic way to extract artist name
        if artist_name not in artists_metadata:
            artists_metadata[artist_name] = []
        artists_metadata[artist_name].append(song_metadata)
        
        # Add to genres metadata
        if 'pop' not in genres_metadata:
            genres_metadata['pop'] = []
        genres_metadata['pop'].append(song_metadata)
    
    # Save metadata
    with open('metadata/songs_metadata.json', 'w') as f:
        json.dump(all_songs, f, indent=4)
    
    with open('metadata/artists_metadata.json', 'w') as f:
        json.dump(artists_metadata, f, indent=4)
    
    with open('metadata/genres_metadata.json', 'w') as f:
        json.dump(genres_metadata, f, indent=4)

    print(f"Downloaded previews and saved metadata for {len(all_songs)} songs.")

if __name__ == "__main__":
    main()