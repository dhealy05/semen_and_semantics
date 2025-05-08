#
from snapshot_loader import *
from reference_embedding import *
from embedding_viz import *
#
import json
import random
from dataclasses import asdict

def make_blank_title_accuracy_file(random_titles, output_file: str = "blank_title_accuracy.json"):

    titles_with_accuracy = []
    for title in random_titles:
        title_dict = asdict(title.metadata)
        title_dict["title_accuracy"] = None
        titles_with_accuracy.append(title_dict)

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(titles_with_accuracy, f, indent=2, default=str)

    print(f"Saved {len(titles_with_accuracy)} random titles to {output_file}")

    return titles_with_accuracy


snapshots_dir = "../snapshots"
loader = SnapshotLoader(snapshots_dir)
collection = loader.load_snapshots()
print(f"Loaded {len(collection)} embeddings from snapshots")

# Get random titles
random_titles = collection.random_sample(n=10)

# Get random titles for year
#groups = collection.group_by_timeperiod('year')
#group = groups['2017']
#random_titles = group.random_sample(n=10)

# Get closest N titles to reference
#references = ReferenceEmbeddings()
#ref_concept = "incest"
#references.make_reference(ref_concept)
#neighbors = collection.get_nearest_neighbors(references.concepts[ref_concept], k=25) # "most ref concept" titles for the year
#neighbors_without_rank = [neighbor[0] for neighbor in neighbors] # make a list
#random_titles = random.sample(neighbors_without_rank, 10) # we will take a subsample

make_blank_title_accuracy_file(random_titles, "blank_title_accuracy.json")
