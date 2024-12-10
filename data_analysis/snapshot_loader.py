from pathlib import Path
from datetime import datetime
import json
#
from embedding_vector import *
from embedding_collection import *

class SnapshotLoader:

    def __init__(self, snapshots_dir: str):
        self.snapshots_dir = Path(snapshots_dir)

    def load_snapshots(self, override_limit=5, exclude_years=[2024]) -> EmbeddingCollection:

        collection = EmbeddingCollection()
        for json_path in self.snapshots_dir.rglob('*.json'):
            date = self._extract_date_from_path(json_path)
            with open(json_path) as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if i > override_limit:
                        continue
                    if 'embedding' in item:  # Ensure item has embedding
                        embedding_dict = item.copy()
                        # Add snapshot date to metadata
                        embedding_dict['timestamp'] = date
                        embedding = EmbeddingVector.from_dict(embedding_dict)
                        if not embedding.metadata.timestamp.year in exclude_years:
                            collection.add_embedding(embedding)

        return collection

    def _extract_date_from_path(self, path: Path) -> datetime:
        # Assuming path structure: snapshots/site/YYYY/MM/YYYYMMDD.json
        parts = path.parts
        year = int(parts[-3])  # Year folder
        month = int(parts[-2])  # Month folder
        day = int(path.stem[:2])  # First two chars of filename
        return datetime(year, month, day)

# Usage:
loader = SnapshotLoader("../snapshots")
collection = loader.load_snapshots()
