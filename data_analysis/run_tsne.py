from snapshot_loader import *
from reference_embedding import *
from embedding_viz import *

def make_references():

    evil_references = ReferenceEmbeddings()
    evil_references.make_reference("woman being raped")
    evil_references.make_reference("incest")
    evil_references.make_reference("torture porn")

    benign_references = ReferenceEmbeddings()
    benign_references.make_reference("people in love")
    benign_references.make_reference("healthy relationships")
    benign_references.make_reference("moral behavior")

    manufacturing_references = ReferenceEmbeddings()
    manufacturing_references.make_reference("airplane factory")
    manufacturing_references.make_reference("blue collar")
    manufacturing_references.make_reference("manufacturing")

    racial_references = ReferenceEmbeddings()
    racial_references.make_reference("african american")
    racial_references.make_reference("latino")
    racial_references.make_reference("asian")

    men_references = ReferenceEmbeddings()
    men_references.make_reference("men digging ditches")
    men_references.make_reference("men lighting laterns")
    men_references.make_reference("men hiking the hills")

    woman_references = ReferenceEmbeddings()
    woman_references.make_reference("woman dancing")
    woman_references.make_reference("woman cooking")
    woman_references.make_reference("woman eating breakfast")

    violence_references = ReferenceEmbeddings()
    violence_references.make_reference("murder")
    violence_references.make_reference("suicide")
    violence_references.make_reference("death")

    haircolor_references = ReferenceEmbeddings()
    haircolor_references.make_reference("brunette")
    haircolor_references.make_reference("blonde")
    haircolor_references.make_reference("redhead")

    pornstar_references = ReferenceEmbeddings()
    pornstar_references.make_reference("Maximus Thrust")
    pornstar_references.make_reference("Ivana Delight")
    pornstar_references.make_reference("Johnny Deep")

    additional_collections = {
        "sexual_violence": evil_references.to_collection(),
        "benign": benign_references.to_collection(),
        "manufacturing": manufacturing_references.to_collection(),
        "racial": racial_references.to_collection(),
        "men": men_references.to_collection(),
        "women": woman_references.to_collection(),
        "violence": violence_references.to_collection(),
        "pornstar": pornstar_references.to_collection(),
        "haircolor": haircolor_references.to_collection()
    }

    return additional_collections

def viz_all(collection):

    viz = EmbeddingVisualization(collection)
    #
    additional_collections = make_references()
    viz.compute_tsne_embedding(additional_collections=additional_collections)

    blank_fig, analysis_results = viz.plot_temporal_tsne(show_years=True, show_collections=None)
    viz.save_fig(blank_fig, "blank")

    for key, value in additional_collections.items():
        key_fig, analysis_results = viz.plot_temporal_tsne(show_years=True, show_collections=[key])
        viz.save_fig(key_fig, key)

    all_fig, analysis_results = viz.plot_temporal_tsne(show_years=True, show_collections=list(additional_collections.keys()))
    viz.save_fig(all_fig, "all")

def analyze(snapshots_dir: str):

    loader = SnapshotLoader(snapshots_dir)
    collection = loader.load_snapshots()
    print(f"Loaded {len(collection)} embeddings from snapshots")

    viz_all(collection)

analyze("../snapshots")
