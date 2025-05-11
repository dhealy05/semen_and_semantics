#
from snapshot_loader import *
from reference_embedding import *
from embedding_viz import *
#
import json
import random
from typing import List
from statistics import mean
from dataclasses import asdict

# see "analysis_results/title_accuracy_logs/readme" for more detail

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

def analyze_title_accuracy(json_files: List[str]) -> None:
    """
    Analyze title accuracy scores across multiple JSON files.
    Calculates the average score for available videos.

    Args:
        json_files: List of paths to JSON files containing title accuracy data
    """
    total_samples = 0
    null_count = 0
    available_scores = []  # List to store all non-null scores

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Count samples in this file
            file_samples = len(data)
            file_null_count = 0
            file_scores = []

            # Count null values and collect scores
            for entry in data:
                if entry['title_accuracy'] is None:
                    file_null_count += 1
                else:
                    file_scores.append(entry['title_accuracy'])

            # Update total counts
            total_samples += file_samples
            null_count += file_null_count
            available_scores.extend(file_scores)

            # Calculate file-specific average
            file_avg = mean(file_scores) if file_scores else 0

            # Print per-file statistics
            print(f"\nFile: {file_path}")
            print(f"Total samples: {file_samples}")
            print(f"Video Not Available (Null): {file_null_count}/{file_samples} Samples")
            print(f"Average Score (for available videos): {file_avg:.2f}")

        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
        except json.JSONDecodeError:
            print(f"Error: File {file_path} contains invalid JSON")

    # Print overall statistics if multiple files were processed
    if len(json_files) > 1:
        overall_avg = mean(available_scores) if available_scores else 0
        print("\nOverall Statistics:")
        print(f"Total samples across all files: {total_samples}")
        print(f"Video Not Available (Null): {null_count}/{total_samples} Samples")
        print(f"Average Score (for available videos): {overall_avg:.2f}")

######### Amend the code below to run your own review ##########

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

#make_blank_title_accuracy_file(random_titles, "blank_title_accuracy.json")

########## Amend the code below to analyze the results #########

#json_files = ["../analysis_results/title_accuracy_logs/title_accuracy_2014.json", "../analysis_results/title_accuracy_logs/incest_title_accuracy.json", "../analysis_results/title_accuracy_logs/overall_title_accuracy_0.json"]
#json_files = ["../analysis_results/title_accuracy_logs/title_accuracy_2022.json"]
#analyze_title_accuracy(json_files)
