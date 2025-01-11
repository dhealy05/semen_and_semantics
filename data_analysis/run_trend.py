from snapshot_loader import *
from reference_embedding import *
from embedding_viz import *

def concept_trend(ref_concept, collection):

    references = ReferenceEmbeddings()
    references.make_reference(ref_concept)

    # semantic search with normalization
    top_matches = collection.get_top_similar_embeddings(
        references.concepts[ref_concept],
        top_percent=0.1  # top 10%
    )
    yearly_stats = top_matches.analyze_yearly_distribution(collection)
    print("\nYearly Analysis:")
    print(f"{'Year':<6} {'Matches':<8} {'Total':<8} {'Rate':<8} {'Normalized':<10}")
    print("-" * 42)
    for year, stats in yearly_stats.items():
        print(f"{year:<6} {stats['raw_count']:<8d} {stats['total_count']:<8d} "
              f"{stats['proportion']:.3f}   {stats['normalized_rate']:.2f}x")
    return yearly_stats

def concept_trends(collection, trends, window_size=2):
    yearly_stats = []
    for trend in trends:
        stats = concept_trend(trend, collection)
        yearly_stats.append(stats)

    viz = EmbeddingVisualization(collection)

    # animated moving average, default to 2 year lookback window
    trends_data = viz.compute_moving_averages(yearly_stats, trends, window_size=window_size)
    viz.animate_trends(trends_data, window_size=window_size)

    # static trend
    fig = viz.plot_concept_trends(yearly_stats, trends)
    viz.save_fig(fig, "trend")

def analyze(snapshots_dir: str):

    loader = SnapshotLoader(snapshots_dir)
    collection = loader.load_snapshots()
    print(f"Loaded {len(collection)} embeddings from snapshots")

    trends = ["rape", "incest"]
    concept_trends(collection, trends)

analyze("../snapshots")
