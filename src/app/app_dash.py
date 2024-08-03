import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pinecone import Pinecone
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi
import hashlib
import json

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

# Function to fetch embeddings from a Pinecone index
def fetch_embeddings(index_name, vector_count, dimensions):
    index = pc.Index(index_name)
    vector_count = index.describe_index_stats()['total_vector_count']
    index_arr = index.query(vector=[0 for _ in range(dimensions)], top_k=vector_count, include_metadata=True)

    vector_ids = [v['id'] for v in index_arr['matches']]
    batch_size = 100  # Adjust batch size as needed
    embeddings = []
    metadata = []

    for i in range(0, len(vector_ids), batch_size):
        batch_ids = vector_ids[i:i + batch_size]
        response = index.fetch(ids=batch_ids)
        batch_embeddings = [v['values'] for v in response['vectors'].values()]
        batch_metadata = [v['metadata'] for v in response['vectors'].values()]
        for id_, meta in zip(batch_ids, batch_metadata):
            meta['id'] = id_
        embeddings.extend(batch_embeddings)
        metadata.extend(batch_metadata)

    return np.array(embeddings), metadata, vector_count

pinecone_ast_name = 'ast-song-embeddings'
pinecone_custom_name = 'custom-song-embeddings'
#  -------------------------------------------------------Fetch song embeddings---------------------------------------------------------
ast_song_embeddings, ast_song_metadata, ast_song_clusters = fetch_embeddings(pinecone_ast_name, 768, 768)
custom_song_embeddings, custom_song_metadata, custom_song_clusters = fetch_embeddings(pinecone_custom_name, 768, 768)

# Connect to the Pinecone index
ast_index = pc.Index(pinecone_ast_name)
custom_index = pc.Index(pinecone_custom_name)

#  -------------------------------------------------------Fetch ast artist embeddings---------------------------------------------------------
# Load artist metadata
with open('../data/metadata/artist_metadata.json') as f:
    artist_metadata = json.load(f)

# Function to fetch embeddings from Pinecone based on song IDs
def fetch_embeddings_by_ids(index, song_ids):
    embeddings = {}
    batch_size = 100  # Adjust batch size as needed
    for i in range(0, len(song_ids), batch_size):
        batch_ids = song_ids[i:i + batch_size]
        response = index.fetch(ids=batch_ids)
        for song_id, vector_data in response['vectors'].items():
            embeddings[song_id] = vector_data['values']
    return embeddings

# Get all song IDs from the artist metadata
all_song_ids = [song['id'] for songs in artist_metadata.values() for song in songs]

# Fetch embeddings for all song IDs
song_embeddings_dict = fetch_embeddings_by_ids(ast_index, all_song_ids)

# Debugging: Print some IDs to verify
#print("Fetched song IDs from Pinecone index:", list(song_embeddings_dict.keys())[:10])

# Calculate average embeddings for each artist
artist_embeddings_dict = {}
for artist, songs in artist_metadata.items():
    embeddings = []
    for song in songs:
        song_id = song['id']
        if song_id in song_embeddings_dict:
            embeddings.append(song_embeddings_dict[song_id])
        
    if embeddings:
        artist_embeddings_dict[artist] = np.mean(embeddings, axis=0)
  

# Convert artist embeddings to a list of tuples (artist, embedding)
artist_embeddings_list = [(artist, embedding) for artist, embedding in artist_embeddings_dict.items()]

# Check if the list is not empty
if not artist_embeddings_list:
    raise ValueError("No artist embeddings found.")

# Separate artist names and embeddings
artist_names, embeddings = zip(*artist_embeddings_list)
ast_artist_embeddings = np.array(embeddings)
ast_artist_clusters = len(artist_metadata.values())
# Transform artist metadata to the new format
transformed_artist_metadata = []
for artist, songs in artist_metadata.items():
    song_names = [song['name'] for song in songs]
    genres = list({song['genre'] for song in songs})
    transformed_artist_metadata.append({
        'genres': genres,
        'name': artist,
        'songs': song_names,
        'id': artist  # Assuming artist name is the unique ID
    })
ast_artist_metadata = transformed_artist_metadata

#  -------------------------------------------------------Fetch ast genre embeddings---------------------------------------------------------
# Load genre metadata
with open('../data/metadata/genres_metadata.json') as f:
    genre_metadata = json.load(f)

# Get all song IDs from the genre metadata
all_genre_song_ids = [song['id'] for songs in genre_metadata.values() for song in songs]

# Fetch embeddings for all song IDs
genre_song_embeddings_dict = fetch_embeddings_by_ids(ast_index, all_genre_song_ids)

# Calculate average embeddings for each genre
genre_embeddings_dict = {}
for genre, songs in genre_metadata.items():
    embeddings = []
    for song in songs:
        song_id = song['id']
        if song_id in genre_song_embeddings_dict:
            embeddings.append(genre_song_embeddings_dict[song_id])
      
    if embeddings:
        genre_embeddings_dict[genre] = np.mean(embeddings, axis=0)


# Convert genre embeddings to a list of tuples (genre, embedding)
genre_embeddings_list = [(genre, embedding) for genre, embedding in genre_embeddings_dict.items()]

# Check if the list is not empty
if not genre_embeddings_list:
    raise ValueError("No genre embeddings found.")

# Separate genre names and embeddings
genre_names, genre_embeddings = zip(*genre_embeddings_list)
ast_genre_embeddings = np.array(genre_embeddings)
ast_genre_clusters = len(genre_metadata.values())
# Transform genre metadata to the new format
transformed_genre_metadata = []
for genre, songs in genre_metadata.items():
    song_names = [song['name'] for song in songs]
    transformed_genre_metadata.append({
        'name': genre,
        'id': genre,  # Assuming genre name is the unique ID
        'songs': song_names
    })
ast_genre_metadata = transformed_genre_metadata

#  -------------------------------------------------------Fetch custom artist embeddings---------------------------------------------------------
# Load artist metadata
with open('../data/metadata/artist_metadata.json') as f:
    artist_metadata = json.load(f)

# Function to fetch embeddings from Pinecone based on song IDs
def fetch_embeddings_by_ids(index, song_ids):
    embeddings = {}
    batch_size = 100  # Adjust batch size as needed
    for i in range(0, len(song_ids), batch_size):
        batch_ids = song_ids[i:i + batch_size]
        response = index.fetch(ids=batch_ids)
        for song_id, vector_data in response['vectors'].items():
            embeddings[song_id] = vector_data['values']
    return embeddings

# Get all song IDs from the artist metadata
all_song_ids = [song['id'] for songs in artist_metadata.values() for song in songs]

# Fetch embeddings for all song IDs
song_embeddings_dict = fetch_embeddings_by_ids(custom_index, all_song_ids)

# Debugging: Print some IDs to verify
#print("Fetched song IDs from Pinecone index:", list(song_embeddings_dict.keys())[:10])

# Calculate average embeddings for each artist
artist_embeddings_dict = {}
for artist, songs in artist_metadata.items():
    embeddings = []
    for song in songs:
        song_id = song['id']
        if song_id in song_embeddings_dict:
            embeddings.append(song_embeddings_dict[song_id])
      
    if embeddings:
        artist_embeddings_dict[artist] = np.mean(embeddings, axis=0)
   

# Convert artist embeddings to a list of tuples (artist, embedding)
artist_embeddings_list = [(artist, embedding) for artist, embedding in artist_embeddings_dict.items()]

# Check if the list is not empty
if not artist_embeddings_list:
    raise ValueError("No artist embeddings found.")

# Separate artist names and embeddings
artist_names, embeddings = zip(*artist_embeddings_list)
custom_artist_embeddings = np.array(embeddings)
custom_artist_clusters = len(artist_metadata.values())
# Transform artist metadata to the new format
transformed_artist_metadata = []
for artist, songs in artist_metadata.items():
    song_names = [song['name'] for song in songs]
    genres = list({song['genre'] for song in songs})
    transformed_artist_metadata.append({
        'genres': genres,
        'name': artist,
        'songs': song_names,
        'id': artist  # Assuming artist name is the unique ID
    })
custom_artist_metadata = transformed_artist_metadata

#  -------------------------------------------------------Fetch custom genre embeddings---------------------------------------------------------
# Load genre metadata
with open('../data/metadata/genres_metadata.json') as f:
    genre_metadata = json.load(f)

# Get all song IDs from the genre metadata
all_genre_song_ids = [song['id'] for songs in genre_metadata.values() for song in songs]

# Fetch embeddings for all song IDs
genre_song_embeddings_dict = fetch_embeddings_by_ids(custom_index, all_genre_song_ids)

# Calculate average embeddings for each genre
genre_embeddings_dict = {}
for genre, songs in genre_metadata.items():
    embeddings = []
    for song in songs:
        song_id = song['id']
        if song_id in genre_song_embeddings_dict:
            embeddings.append(genre_song_embeddings_dict[song_id])
       
    if embeddings:
        genre_embeddings_dict[genre] = np.mean(embeddings, axis=0)


# Convert genre embeddings to a list of tuples (genre, embedding)
genre_embeddings_list = [(genre, embedding) for genre, embedding in genre_embeddings_dict.items()]

# Check if the list is not empty
if not genre_embeddings_list:
    raise ValueError("No genre embeddings found.")

# Separate genre names and embeddings
genre_names, genre_embeddings = zip(*genre_embeddings_list)
custom_genre_embeddings = np.array(genre_embeddings)
custom_genre_clusters = len(genre_metadata.values())
# Transform genre metadata to the new format
transformed_genre_metadata = []
for genre, songs in genre_metadata.items():
    song_names = [song['name'] for song in songs]
    transformed_genre_metadata.append({
        'name': genre,
        'id': genre,  # Assuming genre name is the unique ID
        'songs': song_names
    })
custom_genre_metadata = transformed_genre_metadata


def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2
    new_regions = []
    new_vertices = vor.vertices.tolist()
    all_ridges = construct_ridge_map(vor)
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
        else:
            new_region = reconstruct_infinite_region(p1, vertices, all_ridges, vor, center, radius, new_vertices)
            new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)

def construct_ridge_map(vor):
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    return all_ridges

def reconstruct_infinite_region(p1, vertices, all_ridges, vor, center, radius, new_vertices):
    new_region = [v for v in vertices if v >= 0]
    for p2, v1, v2 in all_ridges[p1]:
        if v2 < 0:
            v1, v2 = v2, v1
        if v1 >= 0:
            continue
        far_point = compute_far_point(p1, p2, v2, vor, center, radius)
        new_region.append(len(new_vertices))
        new_vertices.append(far_point.tolist())
    new_region = sort_region_counterclockwise(new_region, new_vertices)
    return new_region

def compute_far_point(p1, p2, v2, vor, center, radius):
    tangent = vor.points[p2] - vor.points[p1]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    midpoint = vor.points[[p1, p2]].mean(axis=0)
    direction = np.sign(np.dot(midpoint - center, normal)) * normal
    far_point = vor.vertices[v2] + direction * radius
    return far_point

def sort_region_counterclockwise(region, vertices):
    vs = np.asarray([vertices[v] for v in region])
    center = vs.mean(axis=0)
    angles = np.arctan2(vs[:, 1] - center[1], vs[:, 0] - center[0])
    sorted_region = np.array(region)[np.argsort(angles)]
    return sorted_region.tolist()

def generate_color(name, factor=0.25):
    """Generate a pastel color for the given name."""
    # Generate a color based on the hash of the name
    name = np.round(name, 3)
    name = str(name)
    hash_object = hashlib.md5(name.encode())
    hex_dig = hash_object.hexdigest()
    base_color = [int(hex_dig[i:i+2], 16) for i in (0, 2, 4)]

    # Mix the color with white
    pastel_color = [(1 - factor) * c + factor * 255 for c in base_color]
    pastel_color_hex = ''.join(f'{int(c):02x}' for c in pastel_color)

    return '#' + pastel_color_hex

def create_plot(plot_type, model):
    pca = PCA(n_components=2)
    if model == "AST":
        if plot_type == "song":
            reduced_embeddings = pca.fit_transform(ast_song_embeddings)
            clusters = ast_song_clusters
            title = "K-means Clustering on Song Embeddings (PCA-Reduced)"
            metadata = ast_song_metadata
            if len(reduced_embeddings) != len(ast_song_metadata):
                metadata = ast_song_metadata[:len(reduced_embeddings)]
        elif plot_type == "artist":
            reduced_embeddings = pca.fit_transform(ast_artist_embeddings)
            clusters = ast_artist_clusters
            title = "K-means Clustering on Artist Embeddings (PCA-Reduced)"
            metadata = ast_artist_metadata
            if len(reduced_embeddings) != len(ast_artist_metadata):
                metadata = ast_artist_metadata[:len(reduced_embeddings)]
        elif plot_type == "genre":  
            reduced_embeddings = pca.fit_transform(ast_genre_embeddings)
            clusters = ast_genre_clusters
            title = "K-means Clustering on Genre Embeddings (PCA-Reduced)"
            metadata = ast_genre_metadata
            if len(reduced_embeddings) != len(ast_genre_metadata):
                metadata = ast_genre_metadata[:len(reduced_embeddings)]
    elif model == "CUSTOM":
        if plot_type == "song":
            reduced_embeddings = pca.fit_transform(custom_song_embeddings)
            clusters = custom_song_clusters
            title = "K-means Clustering on Song Embeddings (PCA-Reduced)"
            metadata = custom_song_metadata
            if len(reduced_embeddings) != len(custom_song_metadata):
                metadata = custom_song_metadata[:len(reduced_embeddings)]
        elif plot_type == "artist":
            reduced_embeddings = pca.fit_transform(custom_artist_embeddings)
            clusters = custom_artist_clusters
            title = "K-means Clustering on Artist Embeddings (PCA-Reduced)"
            metadata = custom_artist_metadata
            if len(reduced_embeddings) != len(custom_artist_metadata):
                metadata = custom_artist_metadata[:len(reduced_embeddings)]
        elif plot_type == "genre":  
            reduced_embeddings = pca.fit_transform(custom_genre_embeddings)
            clusters = custom_genre_clusters
            title = "K-means Clustering on Genre Embeddings (PCA-Reduced)"
            metadata = custom_genre_metadata
            if len(reduced_embeddings) != len(custom_genre_metadata):
                metadata = custom_genre_metadata[:len(reduced_embeddings)]

    # Ensure lengths match
    
    
    if plot_type == "genre":
        n_clusters = max(clusters // 5, 1)  # For genre embeddings
    else:
        n_clusters = max(clusters // 15, 1)  # For other embeddings
    
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=4, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    data = {
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'name': [m['name'] for m in metadata],
        'cluster': cluster_labels,
        'id': [m['id'] for m in metadata]
    }
    
    if plot_type == "song":
        data['artist'] = [m['artist'] for m in metadata]
        data['preview_url'] = [m['preview_url'] for m in metadata]
        data['album'] = [m['album'] for m in metadata]
        data['cover_image_url'] = [m['cover_image_url'] for m in metadata]
        data['genre'] = [m['genre'] for m in metadata]
    elif plot_type == "artist":
        data['genres'] = [m['genres'] for m in metadata]
        data['songs'] = [m['songs'] for m in metadata]
    elif plot_type == "genre":
        #data['artists'] = [m['artists'] for m in metadata]
        data['songs'] = [m['songs'] for m in metadata]

    
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    # Only create Voronoi regions if there are enough clusters
    if n_clusters >= 4:
        vor = Voronoi(kmeans.cluster_centers_)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        for region, center in zip(regions, kmeans.cluster_centers_):
            polygon = vertices[region]
            color = generate_color(center)
            fig.add_trace(go.Scatter(
                x=polygon[:, 0],
                y=polygon[:, 1],
                fill="toself",
                fillcolor=color,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)')
            ))

    # Add centroids
    fig.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        marker=dict(size=10, color='white', symbol='x'),
        name='Centroids'
    ))

    # Add scatter plot of points
    custom_data = pd.DataFrame()
    if plot_type == "song":
        hover_text = df.apply(lambda row: f"{row['name']} by {row['artist']}", axis=1)
        custom_data['preview_url'] = df['preview_url']
        custom_data['name'] = df['name']
        custom_data['artist'] = df['artist']
        custom_data['album'] = df['album']
        custom_data['cover_image_url'] = df['cover_image_url']
        custom_data['genre'] = df['genre']
    elif plot_type == "artist":
        hover_text = df['name']
        custom_data['name'] = df['name']
        custom_data['genres'] = df['genres']
        custom_data['songs'] = df['songs']
    elif plot_type == "genre":
        hover_text = df['name']
        custom_data['name'] = df['name']
        #custom_data['artists'] = df['artists']
        custom_data['songs'] = df['songs']
    
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(color='black', size=5),
        text=hover_text,
        customdata=custom_data,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        title_y=0.91,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        width=800,
        height=800,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    margin_x = (df['x'].max() - df['x'].min()) * 0.1
    margin_y = (df['y'].max() - df['y'].min()) * 0.1
    
    fig.update_xaxes(minallowed=df['x'].min()-margin_x, maxallowed=df['x'].max()+margin_x)
    fig.update_yaxes(minallowed=df['y'].min()-margin_y, maxallowed=(df['y'].max()+margin_y))
    
    return fig


from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Create the Dash appS
app = Dash(
    external_stylesheets=[dbc.themes.JOURNAL],
    suppress_callback_exceptions=True
)
app.title = "Music Embedding Visualizer"

# Placeholder image URL
placeholder_image_url = "https://www.iforium.com/wp-content/uploads/Placeholder-Image-400.png"

# Layout
app.layout = dbc.Container([
    html.Label('Model:', style={'fontSize': '16px'}),
    dcc.Dropdown(
        id='template-dropdown',
        options=[
            {'label': 'Audio Spectogram Transformer', 'value': 'AST'},
            {'label': 'Custom in-house', 'value': 'CUSTOM'},
        ],
        value='AST',
        style={'marginBottom': '20px'}
    ),
    dcc.Tabs(id="tabs", value='song', children=[
        dcc.Tab(label='Song Embeddings', value='song', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='Artist Embeddings', value='artist', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='Genre Embeddings', value='genre', className='custom-tab', selected_className='custom-tab--selected'),
    ]),
    html.Div(id='tabs-content'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='plot', config={'displayModeBar': False}),
        ], width=7, lg=7),
        dbc.Col([
            dbc.Card([
                dbc.CardImg(src=placeholder_image_url, top=True, id='song-image', style={'maxHeight': '350px', 'objectFit': 'contain'}),
                dbc.CardBody([
                    html.H5("Song Details", className="card-title"),
                    html.P(id='song-name', className="card-text"),
                    html.P(id='artist-name', className="card-text"),
                    html.P(id='album-name', className="card-text"),
                    html.P(id='genre-name', className="card-text"),
                ]),
                dbc.CardFooter([
                    html.Audio(id='audio-player', controls=True, autoPlay=True, style={'width': '100%'})
                ], className="text-center")
            ], style={'height': 'auto', 'margin': '100px 0'})
        ], width=3, lg=3)
    ])
], fluid=True, style={'margin': '20px auto', 'maxWidth': '1500px', 'overflow-x': 'hidden', 'overflow-y': 'hidden'})

@app.callback(
    Output('tabs-content', 'children'),
    Output('plot', 'figure'),
    Input('tabs', 'value'),
    Input('template-dropdown', 'value')
)
def render_content(tab, modelName):
    fig = create_plot(
         plot_type=tab,
         model=modelName,
    )

    return None, fig

@app.callback(
    Output('audio-player', 'src'),
    Output('audio-player', 'style'),
    Output('song-name', 'children'),
    Output('artist-name', 'children'),
    Output('genre-name', 'children'),
    Output('album-name', 'children'),
    Output('song-image', 'src'),
    Input('plot', 'hoverData')
)
def play_audio(hoverData):
    if hoverData and 'customdata' in hoverData['points'][0]:
        preview_url = hoverData['points'][0]['customdata'][0]
        song_name = f"Song: {hoverData['points'][0]['customdata'][1]}"
        artist_name = f"Artist: {hoverData['points'][0]['customdata'][2]}"
        album_name = f"Album: {hoverData['points'][0]['customdata'][3]}"
        genre_name = f"Genre: {hoverData['points'][0]['customdata'][5]}"
        song_image = hoverData['points'][0]['customdata'][4]
        return preview_url, {'display': 'block'}, song_name, artist_name, genre_name, album_name, song_image
    return '', {'display': 'none'}, "Song: ", "Artist: ", "Genre: ", "Album: ", placeholder_image_url

if __name__ == '__main__':
    app.run_server(debug=True)