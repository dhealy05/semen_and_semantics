from typing import List, Dict, Optional, Union
import numpy as np
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Metadata:
    title: str
    url: str
    views: int
    duration: str
    timestamp: datetime

class EmbeddingVector:

    def __init__(self, vector: List[float], metadata: Optional[Metadata] = None):
        self.vector = np.array(vector, dtype=np.float32)
        if metadata is None:
            metadata = Metadata(title='', url='', views=0, duration="", timestamp=datetime.now())
        self.metadata = metadata
        self._normalize()

    def _normalize(self) -> None:
        """Convert to unit vector"""
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

    def cosine_similarity(self, other: 'EmbeddingVector') -> float:
        """Compute cosine similarity with another embedding"""
        return float(np.dot(self.vector, other.vector))

    def length_normalized_cosine_similarity(self, other: 'EmbeddingVector') -> float:
        similarity = float(np.dot(self.vector, other.vector))
        if other.metadata is None:
            return similarity
        title_length = len(other.metadata.title.split())
        return similarity / np.log1p(title_length)  # log to dampen the effect

    def euclidean_distance(self, other: 'EmbeddingVector') -> float:
        """Compute Euclidean distance to another embedding"""
        return float(np.linalg.norm(self.vector - other.vector))

    def manhattan_distance(self, other: 'EmbeddingVector') -> float:
        """Compute Manhattan distance to another embedding"""
        return float(np.sum(np.abs(self.vector - other.vector)))

######## Relative

    def divide(self, other: 'EmbeddingVector') -> 'EmbeddingVector':
        if self.get_dimension() != other.get_dimension():
            raise ValueError("Cannot divide vectors of different dimensions")
        # Check for zeros in denominator
        if np.any(other.vector == 0):
            raise ZeroDivisionError("Division by zero in vector elements")
        # Perform element-wise division
        result_vector = self.vector / other.vector
        # Create new EmbeddingVector with result (no metadata)
        return EmbeddingVector(result_vector.tolist())

    def subtract(self, other: 'EmbeddingVector') -> 'EmbeddingVector':
        if self.get_dimension() != other.get_dimension():
            raise ValueError("Cannot divide vectors of different dimensions")
        # Check for zeros in denominator
        if np.any(other.vector == 0):
            raise ZeroDivisionError("Division by zero in vector elements")
        # Perform element-wise division
        result_vector = self.vector - other.vector
        # Create new EmbeddingVector with result (no metadata)
        return EmbeddingVector(result_vector.tolist())

####### Dicts

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, List[float]]]) -> 'EmbeddingVector':
        """Create EmbeddingVector from dictionary of raw data"""
        metadata = Metadata(
            title=data.get('title', ''),
            url=data.get('url', ''),
            #views=int(data.get('views', '0').replace(',', '')),
            views=data.get('views', '0').replace(',', ''),
            duration=data.get('duration', ''),
            timestamp=data.get('timestamp')
        )
        return cls(data['embedding'], metadata)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'vector': self.vector.tolist(),
            'metadata': {
                'title': self.metadata.title,
                'url': self.metadata.url,
                'views': self.metadata.views,
                'duration': self.metadata.duration,
                'timestamp': self.metadata.timestamp.isoformat()
            } if self.metadata else None
        }

    def get_magnitude(self) -> float:
        """Get vector magnitude"""
        return float(np.linalg.norm(self.vector))

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return len(self.vector)
