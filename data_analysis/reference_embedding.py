from typing import Union
from openai import OpenAI
#
from embedding_vector import *
from embedding_collection import *

class ReferenceEmbeddings:

    def __init__(self):
        self.concepts: Dict[str, EmbeddingVector] = {}

    def make_reference(self, name: str) -> None:
        try:
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=[name]
            )
            vector = EmbeddingVector(response.data[0].embedding)
            self.add_reference(name, vector)
        except Exception as e:
            raise Exception(f"Error creating reference embedding for '{name}': {e}")

    def make_control(self, dim: Optional[int] = None) -> None:

        if dim is None:
            if self.concepts:
                # Use dimension of first concept vector
                dim = len(next(iter(self.concepts.values())).vector)
            else:
                # Default dimension for text-embedding-3-large
                dim = 3072

        # Create random vector and normalize it
        random_vector = np.random.randn(dim)

        vector = EmbeddingVector(random_vector.tolist())
        self.add_reference("control", vector)

    def add_reference(self, name: str, vector: Union[EmbeddingVector, List[float]]) -> None:
        if isinstance(vector, list):
            vector = EmbeddingVector(vector)
        self.concepts[name] = vector

    def get_concept_similarity(self, vector: EmbeddingVector, concept_name: str) -> float:
        if concept_name not in self.concepts:
            raise KeyError(f"Concept '{concept_name}' not found")
        return vector.cosine_similarity(self.concepts[concept_name])

    def get_all_similarities(self, vector: EmbeddingVector) -> Dict[str, float]:
        return {name: self.get_concept_similarity(vector, name)
                for name in self.concepts}

    def get_concept_trajectory(self,
                             collection: EmbeddingCollection,
                             concept_name: str,
                             time_window: str = 'year') -> Dict[str, float]:
        groups = collection.group_by_timeperiod(time_window)
        return {
            period: np.mean([
                self.get_concept_similarity(e, concept_name)
                for e in group
            ])
            for period, group in groups.items()
        }

###### Transform

    def to_collection(self) -> EmbeddingCollection:
        """
        Convert reference embeddings to an EmbeddingCollection.

        Returns:
            EmbeddingCollection: Collection containing all reference embeddings
        """
        collection = EmbeddingCollection()
        for name, vector in self.concepts.items():
            # Create a copy of the vector to avoid reference issues
            embedding = EmbeddingVector(vector.vector.copy())
            # Add metadata with the concept name as title
            embedding.metadata.title = name
            collection.add_embedding(embedding)
        return collection

###### Save / Load

    def save(self, filepath: str) -> None:
        data = {
            name: vector.to_dict()
            for name, vector in self.concepts.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'ReferenceEmbeddings':
        refs = cls()
        with open(filepath) as f:
            data = json.load(f)
        for name, vector_dict in data.items():
            refs.add_reference(name, EmbeddingVector.from_dict(vector_dict))
        return refs
