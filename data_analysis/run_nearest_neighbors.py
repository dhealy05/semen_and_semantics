from snapshot_loader import *
from reference_embedding import *
from embedding_viz import *

def analyze(snapshots_dir: str):

    loader = SnapshotLoader(snapshots_dir)
    collection = loader.load_snapshots()
    print(f"Loaded {len(collection)} embeddings from snapshots")

    groups = collection.group_by_timeperiod('year')

    for year, group in groups.items():
        centroid = group.compute_centroid()
        neighbors = group.get_nearest_neighbors(centroid, k=1)
        for neighbor in neighbors:
            print(neighbor[0].metadata.title)
        print("")
        print("")

analyze("../snapshots")
