from snapshot_loader import *
from reference_embedding import *
from embedding_viz import *

def viz_all(collection):

    viz = EmbeddingVisualization(collection)
    #
    fig, analysis = viz.plot_year_centroids_similarity()
    viz.save_fig(fig, "centroid")

def analyze(snapshots_dir: str):

    loader = SnapshotLoader(snapshots_dir)
    collection = loader.load_snapshots()
    print(f"Loaded {len(collection)} embeddings from snapshots")
    viz_all(collection)

analyze("../snapshots")
