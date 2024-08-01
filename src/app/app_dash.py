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
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

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

# Fetch song embeddings
song_embeddings, song_metadata, song_clusters = fetch_embeddings('song-embeddings', 768, 768)

# Fetch artist embeddings
artist_embeddings, artist_metadata, artist_clusters = fetch_embeddings('artist-embeddings', 768, 768)

# Fetch genre embeddings
genre_embeddings, genre_metadata, genre_clusters = fetch_embeddings('genre-embeddings', 768, 768)

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
    hash_object = hashlib.md5(name.encode())
    hex_dig = hash_object.hexdigest()
    base_color = [int(hex_dig[i:i+2], 16) for i in (0, 2, 4)]
    pastel_color = [(1 - factor) * c + factor * 255 for c in base_color]
    pastel_color_hex = ''.join(f'{int(c):02x}' for c in pastel_color)
    return '#' + pastel_color_hex

def create_plot(embeddings, metadata, clusters, title, plot_type):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    if plot_type == "genre":
        n_clusters = max(clusters // 5, 1)  # For genre embeddings
    else:
        n_clusters = max(clusters // 20, 1)  # For other embeddings
    
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
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    # Only create Voronoi regions if there are enough clusters
    if n_clusters >= 4:
        vor = Voronoi(kmeans.cluster_centers_)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        for region, center in zip(regions, kmeans.cluster_centers_):
            polygon = vertices[region]
            color = generate_color(str(center))
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
    if plot_type == "song":
        hover_text = df.apply(lambda row: f"{row['name']} by {row['artist']}", axis=1)
        custom_data = df['preview_url']
    else:
        hover_text = df['name']
        custom_data = None
    
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
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        width=800,
        height=800,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(minallowed=df['x'].min()-1, maxallowed=df['x'].max()+1)
    fig.update_yaxes(minallowed=df['y'].min()-1, maxallowed=(df['y'].max()+1))
    
    return fig

# Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Song Embeddings Visualization"),
    dcc.Tabs(id="tabs", value='song', children=[
        dcc.Tab(label='Song Embeddings', value='song'),
        dcc.Tab(label='Artist Embeddings', value='artist'),
        dcc.Tab(label='Genre Embeddings', value='genre'),
    ]),
    html.Div(id='tabs-content'),
    dcc.Graph(id='plot', style={'width': '800px', 'height': '800px'}),
    html.Audio(id='audio-player', controls=True, autoPlay=True)
])

@app.callback(
    Output('tabs-content', 'children'),
    Output('plot', 'figure'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'song':
        fig = create_plot(song_embeddings, song_metadata, song_clusters, "K-means Clustering on Song Embeddings (PCA-Reduced)", "song")
    elif tab == 'artist':
        fig = create_plot(artist_embeddings, artist_metadata, artist_clusters, "K-means Clustering on Artist Embeddings (PCA-Reduced)", "artist")
    elif tab == 'genre':
        fig = create_plot(genre_embeddings, genre_metadata, genre_clusters, "K-means Clustering on Genre Embeddings (PCA-Reduced)", "genre")
    else:
        fig = {}
    return None, fig

@app.callback(
    Output('audio-player', 'src'),
    Output('audio-player', 'style'),
    Input('plot', 'hoverData')
)
def play_audio(hoverData):
    if hoverData and 'customdata' in hoverData['points'][0]:
        preview_url = hoverData['points'][0]['customdata']
        return preview_url, {'display': 'block'}
    return '', {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True)
