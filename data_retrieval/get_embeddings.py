from pathlib import Path
import json
from typing import Dict, List
import os
from openai import OpenAI
from tqdm import tqdm

class EmbeddingProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.client = OpenAI()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

    def process_json_file(self, json_path: Path) -> None:
        try:
            # Read existing JSON
            with open(json_path, 'r') as f:
                videos = json.load(f)

            # Get titles that need embeddings
            titles_to_process = []
            indices_to_update = []

            for i, video in enumerate(videos):
                #if 'embedding' not in video and video.get('title'):
                if video.get('title'):
                    titles_to_process.append(video['title'])
                    indices_to_update.append(i)

            if not titles_to_process:
                return

            # Get embeddings in batch
            embeddings = self.get_embeddings(titles_to_process)

            # Update videos with embeddings
            for idx, embedding in zip(indices_to_update, embeddings):
                videos[idx]['embedding'] = embedding

            # Save updated JSON
            with open(json_path, 'w') as f:
                json.dump(videos, f, indent=2)

            print(f"Processed {len(titles_to_process)} videos in {json_path}")

        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    def process_all_files(self):
        json_files = list(self.base_dir.rglob('*.json'))
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            self.process_json_file(json_file)

def main():
    base_dir = "../snapshots"
    processor = EmbeddingProcessor(base_dir)
    processor.process_all_files()

if __name__ == "__main__":
    main()
