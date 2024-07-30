import os
import json
import subprocess
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import collections

# Spotify API credentials
SPOTIPY_CLIENT_ID = '11419dac9f88424c93dd910d7d82a6bc'
SPOTIPY_CLIENT_SECRET = 'f29c523eb7fe44f686f59e5e9938c87a'

# Function to get the top genres by analyzing top tracks globally
def get_top_genres(sp):
    genres = sp.recommendation_genre_seeds()['genres']
    return genres # Limiting to top 62 genres

# Function to get the top songs for a genre
def get_top_songs_for_genre(sp, genre):
    results = sp.search(q=f'genre:{genre}', type='track', limit=15)
    return [(track['name'], track['artists'][0]['name'], genre) for track in results['tracks']['items']]

# Function to write popular songs to a text file, only if they don't already exist
def write_popular_songs_to_file(file_path, popular_songs):
    existing_songs = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_songs.update(line.strip() for line in file)
    
    with open(file_path, 'a', encoding='utf-8') as file:
        for song, artist, genre in popular_songs:
            if song not in existing_songs:
                file.write(f"{song} - {artist} - {genre}\n")
                existing_songs.add(song)

# Function to read popular songs from a text file
def read_popular_songs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip().split(" - ") for line in lines]

# Download preview using ytmdl and convert to 16kHz mono MP3
def download_and_convert(song_name, output_dir='previews'):
    temp_output_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Sanitize song name for file system
    safe_song_name = song_name.replace('"', '').replace("'", "").replace(":", "")
    
    try:
        # Download the song using ytmdl to the temporary directory
        command_download = f'ytmdl "{safe_song_name}" -q -o {temp_output_dir} --ignore-chapters'
        subprocess.run(command_download, shell=True, check=True)
        
        # Find the downloaded file (assuming there's only one file in the temp directory)
        downloaded_files = os.listdir(temp_output_dir)
        if not downloaded_files:
            raise FileNotFoundError(f"No files found in {temp_output_dir} after download.")
        
        temp_output = os.path.join(temp_output_dir, downloaded_files[0])
        final_output = os.path.join(output_dir, f"{safe_song_name.replace(' ', '_')}.mp3")
        
        # Convert the audio to 16kHz mono MP3 using ffmpeg
        command_convert = f'ffmpeg -i "{temp_output}" -ac 1 -ar 16000 "{final_output}"'
        subprocess.run(command_convert, shell=True, check=True)
        
        # Remove the temporary files
        for file in downloaded_files:
            os.remove(os.path.join(temp_output_dir, file))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error downloading or converting {song_name}: {e}")

# Main function
def main():
<<<<<<< HEAD
    # Spotify authentication
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

    # Get the top 62 genres
    top_genres = get_top_genres(sp)

    # Get the top 40 songs for each genre
    popular_songs = []
    for genre in tqdm(top_genres, desc="Fetching top songs for genres"):
        popular_songs.extend(get_top_songs_for_genre(sp, genre))

    # Write popular songs to the text file only if they don't already exist
    popular_songs_file = 'popular_songs.txt'
    write_popular_songs_to_file(popular_songs_file, popular_songs)

=======
    # File containing the list of popular songs in the pop genre
    popular_songs_file = './popular_songs.txt'
    
>>>>>>> 9c7968056b8a3d1458a48f292f1de44cf737206a
    # Read popular songs from the text file
    popular_songs = read_popular_songs(popular_songs_file)
    
    all_songs = []
    artists_metadata = {}
    genres_metadata = {genre: [] for genre in top_genres}
    
    # Create directories if they don't exist
    os.makedirs('previews', exist_ok=True)
    os.makedirs('metadata', exist_ok=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(download_and_convert, f"{song_name} - {artist}") for song_name, artist, genre in popular_songs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading and converting"):
            future.result()

    # Save metadata
    for song_name, artist, genre in popular_songs:
        song_metadata = {
            'name': song_name,
            'artist': artist,
            'genres': genre
        }
        all_songs.append(song_metadata)
        
        # Add to artists metadata
        if artist not in artists_metadata:
            artists_metadata[artist] = []
        artists_metadata[artist].append(song_metadata)
        
        # Add to genres metadata
        genres_metadata[genre].append(song_metadata)
    
    with open('metadata/songs_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(all_songs, f, indent=4)
    
    with open('metadata/artists_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(artists_metadata, f, indent=4)
    
    with open('metadata/genres_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(genres_metadata, f, indent=4)

    print(f"Downloaded previews and saved metadata for {len(all_songs)} songs.")

if __name__ == "__main__":
    main()