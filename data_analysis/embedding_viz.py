import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from typing import Dict, Tuple, Optional, List
import os

class EmbeddingVisualization:

    def __init__(self, collection):
        self.collection = collection
        self.cached_tsne_coords = None
        self.cached_labels = None
        self.cached_vectors = None

##### Concepts

    def plot_concept_trends(self,
                           yearly_stats_array: List[Dict[str, Dict[str, float]]],
                           labels: List[str],
                           metric: str = 'normalized_rate',
                           figsize: tuple = (12, 6),
                           title: str = 'Concept Trends Over Time'):
        """
        Plot multiple concept trends over time.

        Args:
            yearly_stats_array: List of yearly_stats dictionaries for different concepts
            labels: List of labels for each concept trend
            metric: Which metric to plot ('normalized_rate', 'proportion', 'raw_count')
            figsize: Figure size as (width, height)
            title: Plot title
        """
        fig = plt.figure(figsize=figsize)

        # For each concept's yearly stats
        for yearly_stats, label in zip(yearly_stats_array, labels):
            # Extract years and values
            years = [int(year) for year in yearly_stats.keys()]
            values = [stats[metric] for stats in yearly_stats.values()]

            # Plot points and line
            plt.plot(years, values, 'o-', label=label, linewidth=2, markersize=6)

            # Optional: Add smoothed trend line
            #z = np.polyfit(years, values, 3)
            #p = np.poly1d(z)
            #x_smooth = np.linspace(min(years), max(years), 100)
            #plt.plot(x_smooth, p(x_smooth), '--', alpha=0.5)

        # Customize plot
        plt.grid(True, alpha=0.3)
        plt.xlabel('Year')

        # Set y-axis label based on metric
        metric_labels = {
            'normalized_rate': 'Normalized Rate (1.0 = baseline)',
            'proportion': 'Proportion',
            'raw_count': 'Number of Matches'
        }
        plt.ylabel(metric_labels.get(metric, metric))

        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add horizontal line at y=1.0 for normalized rate
        if metric == 'normalized_rate':
            plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

        plt.tight_layout()
        #plt.gcf()
        return fig

##### Heatmap

    def plot_year_centroids_similarity(self,
                                     figsize: Tuple[int, int] = (10, 8),
                                     cmap: str = 'RdYlGn') -> Tuple[plt.Figure, Dict]:
        """
        Visualize similarity matrix between year centroids.
        Args:
            figsize: Tuple specifying figure dimensions
            cmap: Colormap to use for heatmap
        Returns:
            Tuple containing:
                - matplotlib Figure object
                - Dict containing analysis results
        """
        # Group embeddings by year and compute centroids
        year_groups = self.collection.group_by_timeperiod('year')
        years = sorted(year_groups.keys())
        # Compute centroid for each year
        year_centroids = {}
        for year in years:
            year_centroids[year] = year_groups[year].compute_centroid()

        # Create similarity matrix
        n_years = len(years)
        similarity_matrix = np.zeros((n_years, n_years))
        for i, year1 in enumerate(years):
            for j, year2 in enumerate(years):
                similarity = year_centroids[year1].cosine_similarity(year_centroids[year2])
                similarity_matrix[i, j] = similarity

        # Normalize similarity matrix to [0,1] range
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        sim_min = similarity_matrix[mask].min()
        sim_max = similarity_matrix[mask].max()
        normalized_matrix = (similarity_matrix - sim_min) / (sim_max - sim_min)
        np.fill_diagonal(normalized_matrix, 1)

        # Create heatmap
        fig = plt.figure(figsize=figsize)
        sns.heatmap(
            normalized_matrix,
            xticklabels=years,
            yticklabels=years,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.2f'
        )
        plt.title('Normalized Similarity Matrix of Year Centroids')
        plt.xlabel('Year')
        plt.ylabel('Year')
        plt.tight_layout()

        # Compute analysis results
        mask = ~np.eye(n_years, dtype=bool)
        max_sim = similarity_matrix[mask].max()
        min_sim = similarity_matrix[mask].min()
        max_idx = np.where((similarity_matrix == max_sim) & mask)
        min_idx = np.where((similarity_matrix == min_sim) & mask)

        analysis_results = {
            "most_similar": {
                "years": (years[max_idx[0][0]], years[max_idx[1][0]]),
                "similarity": float(max_sim),
                "normalized": 1.0
            },
            "least_similar": {
                "years": (years[min_idx[0][0]], years[min_idx[1][0]]),
                "similarity": float(min_sim),
                "normalized": 0.0
            },
            "similarity_matrix": similarity_matrix.tolist(),
            "normalized_matrix": normalized_matrix.tolist()
        }

        return fig, analysis_results

##### t-SNE

    def compute_tsne_embedding(self,
                             additional_collections: Optional[Dict[str, 'EmbeddingCollection']] = None,
                             perplexity: int = 5,
                             n_iter: int = 2000,
                             random_state: int = 42) -> None:
        """
        Precompute t-SNE embeddings for all points to establish stable coordinate space.
        """
        # Group embeddings by year
        year_groups = self.collection.group_by_timeperiod('year')
        years = sorted(year_groups.keys())

        # Prepare data arrays
        all_vectors = []
        all_labels = []

        # Add year centroids first
        for year in years:
            centroid = year_groups[year].compute_centroid()
            all_vectors.append(centroid.vector)
            all_labels.append(year)

        # Add additional collections
        if additional_collections:
            for label, collection in additional_collections.items():
                for embedding in collection.embeddings:
                    all_vectors.append(embedding.vector)
                    all_labels.append(label)

        vectors_array = np.array(all_vectors)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            early_exaggeration=4,
            learning_rate=200,
            n_iter=n_iter,
            random_state=random_state,
            metric='cosine'
        )

        self.cached_tsne_coords = tsne.fit_transform(vectors_array)
        self.cached_labels = all_labels
        self.cached_vectors = vectors_array

    def plot_temporal_tsne(self,
                          show_years: bool = True,
                          show_collections: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, Dict]:
        """
        Plot t-SNE visualization using cached coordinates with selective display of points.
        """
        if self.cached_tsne_coords is None:
            raise ValueError("Must call compute_tsne_embedding() first")

        # Create mask for points to display
        display_mask = np.zeros(len(self.cached_labels), dtype=bool)

        # Get indices for years
        year_indices = [i for i, label in enumerate(self.cached_labels)
                       if isinstance(label, str) and label.isdigit()]
        if show_years:
            display_mask[year_indices] = True

        # Handle additional collections
        if show_collections:
            for collection_label in show_collections:
                collection_indices = [i for i, label in enumerate(self.cached_labels)
                                   if label == collection_label]
                display_mask[collection_indices] = True

        # Get displayed points
        displayed_coords = self.cached_tsne_coords[display_mask]
        displayed_labels = np.array(self.cached_labels)[display_mask]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Assign colors
        colors = np.zeros(len(displayed_labels))
        next_color_id = 0

        if show_years:
            for i, label in enumerate(displayed_labels):
                if isinstance(label, str) and label.isdigit():
                    if label <= "2009":
                        colors[i] = 0
                    elif label <= "2016":
                        colors[i] = 1
                    else:
                        colors[i] = 2
            next_color_id = 3

        # Assign colors for additional collections
        if show_collections:
            for collection_label in show_collections:
                collection_mask = displayed_labels == collection_label
                colors[collection_mask] = next_color_id
                next_color_id += 1

        # Set consistent axis limits
        all_coords = self.cached_tsne_coords
        padding = 0.1
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

        scatter = ax.scatter(displayed_coords[:, 0], displayed_coords[:, 1],
                           c=colors, cmap='Set1', s=100)

        # Add labels
        for i, label in enumerate(displayed_labels):
            ax.annotate(str(label),
                       (displayed_coords[i, 0], displayed_coords[i, 1]),
                       xytext=(5, 5), textcoords='offset points')

        # Simple legend
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=plt.cm.Set1(i/(next_color_id-1)),
                                label=f'Cluster {i+1}',
                                markersize=10)
                         for i in range(next_color_id)]
        ax.legend(handles=legend_elements)
        ax.set_title('t-SNE Visualization')
        plt.tight_layout()

        return fig, {
            "coordinates": displayed_coords.tolist(),
            "labels": displayed_labels.tolist(),
            "colors": colors.tolist(),
            "axis_limits": {
                "x": [float(x_min), float(x_max)],
                "y": [float(y_min), float(y_max)]
            }
        }

#### Save

    def save_fig(self,
                  fig,
                  filename: str = "viz",
                  output_dir: str = "../analysis_results",
                  dpi: int = 300) -> None:
        fig.savefig(
            os.path.join(output_dir, f"{filename}.png"),
            dpi=dpi,
            bbox_inches='tight'
        )
        plt.close(fig)
