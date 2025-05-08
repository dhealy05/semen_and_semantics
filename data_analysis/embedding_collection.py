from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import random
#
from embedding_vector import *

class EmbeddingCollection:

###### Init

    def __init__(self):
        self.embeddings: List[EmbeddingVector] = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._needs_update = True

    def add_embedding(self, embedding: EmbeddingVector) -> None:
        """Add single embedding to collection"""
        self.embeddings.append(embedding)
        self._needs_update = True

    def add_embeddings(self, embeddings: List[EmbeddingVector]) -> None:
        """Add multiple embeddings to collection"""
        self.embeddings.extend(embeddings)
        self._needs_update = True

    def _update_matrix(self) -> None:
        """Update the internal numpy matrix for efficient computation"""
        if self._needs_update:
            self._embedding_matrix = np.vstack([e.vector for e in self.embeddings])
            self._needs_update = False

##### Operations

    def get_nearest_neighbors(self, query: EmbeddingVector, k: int = 5) -> List[Tuple[EmbeddingVector, float]]:
        """Find k nearest neighbors to query vector"""
        self._update_matrix()
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(self._embedding_matrix)
        distances, indices = nbrs.kneighbors([query.vector])

        return [(self.embeddings[idx], dist)
                for idx, dist in zip(indices[0], distances[0])]

    def random_sample(self, n: int = 5):
        return random.sample(self.embeddings, n)

    def get_average_title_length(self) -> float:

        total_length = sum(
            len(embedding.metadata.title)
            for embedding in self.embeddings
            if embedding.metadata and embedding.metadata.title
        )

        valid_titles = sum(
            1 for embedding in self.embeddings
            if embedding.metadata and embedding.metadata.title
        )

        return total_length / valid_titles if valid_titles > 0 else 0.0

    def compute_centroid(self) -> EmbeddingVector:
        """
        Compute centroid with outlier removal and equal time-based weighting
        """
        self._update_matrix()

        # Group embeddings by timestamp (e.g., by day)
        from collections import defaultdict
        daily_groups = defaultdict(list)
        for i, emb in enumerate(self.embeddings):
            day_key = emb.metadata.timestamp.strftime('%Y-%m-%d')
            daily_groups[day_key].append(i)

        # Compute per-day centroids first
        daily_centroids = []
        for indices in daily_groups.values():
            day_vectors = self._embedding_matrix[indices]
            day_centroid = np.mean(day_vectors, axis=0)
            daily_centroids.append(day_centroid)

        # Then compute final centroid from daily centroids
        final_centroid = np.mean(daily_centroids, axis=0)
        return EmbeddingVector(final_centroid.tolist())

    #def compute_centroid(self) -> EmbeddingVector:
    #    """Compute centroid of all embeddings"""
    #    self._update_matrix()
    #    centroid = np.mean(self._embedding_matrix, axis=0)
    #    return EmbeddingVector(centroid.tolist())

    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise similarity matrix"""
        self._update_matrix()
        return np.dot(self._embedding_matrix, self._embedding_matrix.T)

###### Search

    def calculate_probability(self, query) -> float:

        total_vectors = len(self.embeddings)
        if total_vectors == 0:
            return 0.0

        # Exact string match mode - look for query in titles/content
        matching_vectors = sum(
            1 for embedding in self.embeddings
            if (embedding.metadata and
                (embedding.metadata.title and query.lower() in embedding.metadata.title.lower()))
        )

        return matching_vectors / total_vectors

    def get_top_similar_embeddings(self, query_vector: EmbeddingVector, top_percent: float = 0.1) -> 'EmbeddingCollection':
        """
        Get a new EmbeddingCollection containing the top X% most similar embeddings based on
        normalized similarity scores.

        Args:
            query_vector: Vector representing the concept to measure
            top_percent: Percentage of most similar embeddings to consider (0.1 = top 10%)

        Returns:
            New EmbeddingCollection containing only the top matches
        """
        # Calculate similarities for all embeddings
        similarities = np.array([
            emb.cosine_similarity(query_vector)
            for emb in self.embeddings
        ])

        # Normalize similarities (z-score)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        normalized_similarities = (similarities - mean_sim) / std_sim

        # Calculate threshold for top X percent
        num_top = int(len(similarities) * top_percent)
        if num_top == 0:
            return EmbeddingCollection()

        # Get indices of top matches
        top_indices = np.argpartition(normalized_similarities, -num_top)[-num_top:]

        # Create new collection with top matches
        top_collection = EmbeddingCollection()
        for idx in top_indices:
            top_collection.add_embedding(self.embeddings[idx])

        return top_collection

    def analyze_yearly_distribution(self, full_collection: Optional['EmbeddingCollection'] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze the distribution of embeddings across years, accounting for different year sizes.

        Args:
            full_collection: The complete collection to use as a baseline for normalization
                            If None, will only provide raw counts and proportions

        Returns:
            Dictionary mapping years to their statistics including:
            - raw_count: Number of matches in that year
            - total_count: Total number of embeddings in that year
            - proportion: Proportion of matches relative to year size
            - normalized_rate: Proportion normalized against expected rate
        """
        if full_collection is None:
            raise ValueError("full_collection must be provided for proper normalization")

        # Get total counts per year from full collection first
        total_year_counts = defaultdict(int)
        for embedding in full_collection.embeddings:
            year = embedding.metadata.timestamp.strftime('%Y')
            total_year_counts[year] += 1

        # Count matches per year in this collection
        match_year_counts = defaultdict(int)
        for embedding in self.embeddings:
            year = embedding.metadata.timestamp.strftime('%Y')
            match_year_counts[year] += 1

        # Calculate expected rate
        total_matches = len(self.embeddings)
        total_embeddings = len(full_collection.embeddings)
        expected_rate = total_matches / total_embeddings if total_embeddings > 0 else 0

        # Calculate statistics for each year that exists in the full collection
        stats = {}
        for year, total_in_year in sorted(total_year_counts.items()):
            matches_in_year = match_year_counts[year]
            actual_rate = matches_in_year / total_in_year if total_in_year > 0 else 0

            stats[year] = {
                'raw_count': matches_in_year,
                'total_count': total_in_year,
                'proportion': actual_rate,
                'normalized_rate': actual_rate / expected_rate if expected_rate > 0 else 0
            }

        return stats

###### Time Analysis

    def get_embeddings_by_date_range(self, start: datetime, end: datetime) -> 'EmbeddingCollection':
        """Get subset of embeddings within date range"""
        filtered = [e for e in self.embeddings
                   if start <= e.metadata.timestamp <= end]
        collection = EmbeddingCollection()
        collection.add_embeddings(filtered)
        return collection

    def group_by_timeperiod(self, period: str = 'year', exclude_years: int = [2024]) -> Dict[str, 'EmbeddingCollection']:
        """Group embeddings by time period (year/month/week)"""
        groups = defaultdict(EmbeddingCollection)

        for embedding in self.embeddings:
            if embedding.metadata.timestamp.year in exclude_years:
                continue
            if period == 'year':
                key = embedding.metadata.timestamp.strftime('%Y')
            elif period == 'month':
                key = embedding.metadata.timestamp.strftime('%Y-%m')
            elif period == 'week':
                key = embedding.metadata.timestamp.strftime('%Y-W%W')
            groups[key].add_embedding(embedding)

        return dict(sorted(groups.items(), key=lambda x: x[0]))

    def analyze_date_coverage(self):
        # Get all timestamps
        timestamps = [e.metadata.timestamp for e in self.embeddings]

        # Basic date range
        start_date = min(timestamps)
        end_date = max(timestamps)

        # Group by year to see distribution
        year_groups = self.group_by_timeperiod('year')

        # Detailed analysis
        print("=== Date Coverage Analysis ===")
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Total embeddings: {len(self)}")
        print("\nDistribution by year:")

        for year, group in sorted(year_groups.items()):
            # Get month coverage for each year
            months_in_year = set(e.metadata.timestamp.strftime('%m') for e in group.embeddings)
            print(f"{year}: {len(group)} embeddings (months: {sorted(months_in_year)})")

        # Check for gaps
        all_years = set(int(year) for year in year_groups.keys())
        expected_years = set(range(min(all_years), max(all_years) + 1))
        missing_years = expected_years - all_years

        if missing_years:
            print("\nMissing years:", sorted(missing_years))

        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_embeddings": len(self),
            "year_distribution": {year: len(group) for year, group in year_groups.items()},
            "missing_years": sorted(missing_years)
        }

###### Save / Load

    def save(self, filepath: str) -> None:
        """Save collection to file"""
        data = {
            'embeddings': [e.to_dict() for e in self.embeddings]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'EmbeddingCollection':
        """Load collection from file"""
        with open(filepath) as f:
            data = json.load(f)

        collection = cls()
        for embedding_dict in data['embeddings']:
            embedding = EmbeddingVector.from_dict(embedding_dict)
            collection.add_embedding(embedding)

        return collection

##### Helpers

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> EmbeddingVector:
        return self.embeddings[idx]
