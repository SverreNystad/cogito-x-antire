import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import requests
from sklearn.mixture import GaussianMixture
import tiktoken

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

@dataclass
class Node:
    text: str
    embedding: List[float]
    children: List[int]
    level: int
    source: Optional[str] = None

class RaptorIndex:
    def __init__(self, embedding_model: str = "openai/text-embedding-3-small",
                 summary_model: str = "openai/gpt-4o-mini"):
        self.embedding_model = embedding_model
        self.summary_model = summary_model
        self.nodes: List[Node] = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _call_openrouter(self, endpoint: str, payload: dict) -> dict:
        response = requests.post(
            f"{OPENROUTER_BASE}/{endpoint}",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def _get_embedding(self, text: str) -> List[float]:
        result = self._call_openrouter("embeddings", {
            "model": self.embedding_model,
            "input": text
        })
        return result["data"][0]["embedding"]

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        result = self._call_openrouter("embeddings", {
            "model": self.embedding_model,
            "input": texts
        })
        return [d["embedding"] for d in sorted(result["data"], key=lambda x: x["index"])]

    def _summarize(self, texts: List[str]) -> str:
        combined = "\n\n---\n\n".join(texts)
        result = self._call_openrouter("chat/completions", {
            "model": self.summary_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries. Summarize the following text chunks into a coherent summary that captures the key information."},
                {"role": "user", "content": f"Summarize the following:\n\n{combined}"}
            ],
            "max_tokens": 1000
        })
        return result["choices"][0]["message"]["content"]

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap if end < len(tokens) else end
        return chunks

    def _cluster_embeddings(self, embeddings: np.ndarray, max_clusters: int = 10,
                             min_clusters: int = 2, target_cluster_size: int = 20) -> np.ndarray:
        n_samples = len(embeddings)
        if n_samples <= 1:
            return np.zeros(n_samples, dtype=int)

        # Calculate target number of clusters based on desired cluster size
        target_k = max(min_clusters, n_samples // target_cluster_size)
        max_k = min(max_clusters, n_samples // 2)  # At least 2 items per cluster
        min_k = min(min_clusters, max_k)

        best_k = target_k
        best_bic = float('inf')

        # Search around target_k for best BIC
        for k in range(min_k, max_k + 1):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except:
                break

        # Ensure we have at least min_clusters
        best_k = max(best_k, min_k)

        gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=3)
        return gmm.fit_predict(embeddings)

    def index_documents(self, doc_paths: List[Path], max_levels: int = 3):
        print(f"Indexing {len(doc_paths)} documents...")

        # Level 0: Create leaf nodes from chunks
        all_chunks = []
        chunk_sources = []
        for path in doc_paths:
            text = path.read_text()
            chunks = self._chunk_text(text)
            all_chunks.extend(chunks)
            chunk_sources.extend([path.name] * len(chunks))
            print(f"  {path.name}: {len(chunks)} chunks")

        print(f"\nTotal chunks: {len(all_chunks)}")
        print("Getting embeddings for leaf nodes...")

        # Batch embeddings
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            embeddings = self._get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
            print(f"  Embedded {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")

        # Create leaf nodes
        leaf_indices = []
        for i, (chunk, emb, source) in enumerate(zip(all_chunks, all_embeddings, chunk_sources)):
            node = Node(text=chunk, embedding=emb, children=[], level=0, source=source)
            self.nodes.append(node)
            leaf_indices.append(len(self.nodes) - 1)

        # Build tree levels
        current_level_indices = leaf_indices
        for level in range(1, max_levels + 1):
            if len(current_level_indices) <= 1:
                break

            print(f"\nBuilding level {level}...")
            embeddings = np.array([self.nodes[i].embedding for i in current_level_indices])
            clusters = self._cluster_embeddings(embeddings)
            n_clusters = len(set(clusters))
            print(f"  Created {n_clusters} clusters")

            next_level_indices = []
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_indices = [current_level_indices[i] for i in range(len(current_level_indices)) if cluster_mask[i]]

                if len(cluster_indices) == 0:
                    continue

                # Summarize cluster
                cluster_texts = [self.nodes[i].text for i in cluster_indices]
                summary = self._summarize(cluster_texts[:10])  # Limit to avoid token limits

                # Get embedding for summary
                summary_embedding = self._get_embedding(summary)

                # Create parent node
                node = Node(
                    text=summary,
                    embedding=summary_embedding,
                    children=cluster_indices,
                    level=level
                )
                self.nodes.append(node)
                next_level_indices.append(len(self.nodes) - 1)
                print(f"  Cluster {cluster_id}: {len(cluster_indices)} children")

            current_level_indices = next_level_indices

        print(f"\nIndex complete: {len(self.nodes)} total nodes")

    def save(self, path: Path):
        data = {
            "embedding_model": self.embedding_model,
            "summary_model": self.summary_model,
            "nodes": [asdict(n) for n in self.nodes]
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Saved index to {path}")

    def load(self, path: Path):
        data = json.loads(path.read_text())
        self.embedding_model = data["embedding_model"]
        self.summary_model = data["summary_model"]
        self.nodes = [Node(**n) for n in data["nodes"]]
        print(f"Loaded index with {len(self.nodes)} nodes")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Flat search across all nodes (original method)."""
        query_embedding = np.array(self._get_embedding(query))

        # Search across all levels
        results = []
        for i, node in enumerate(self.nodes):
            node_embedding = np.array(node.embedding)
            similarity = self._cosine_similarity(query_embedding, node_embedding)
            results.append({
                "index": i,
                "text": node.text,
                "level": node.level,
                "source": node.source,
                "similarity": similarity
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def tree_search(self, query: str, top_k: int = 5, branch_factor: int = 3) -> List[Dict]:
        """
        Hierarchical RAPTOR tree traversal.

        1. Start at highest level (root summaries)
        2. Select top-k most relevant branches
        3. Recursively explore children
        4. Return best leaf nodes
        """
        query_embedding = np.array(self._get_embedding(query))

        # Find max level (root)
        max_level = max(n.level for n in self.nodes)

        # Get root nodes
        root_indices = [i for i, n in enumerate(self.nodes) if n.level == max_level]

        # If only one root or no tree structure, fall back to flat search
        if max_level == 0:
            return self.search(query, top_k)

        # Track visited nodes and collect leaves
        collected_leaves = []

        def traverse(node_indices: List[int], depth: int = 0):
            if not node_indices:
                return

            # Score all nodes at this level
            scored = []
            for idx in node_indices:
                node = self.nodes[idx]
                sim = self._cosine_similarity(query_embedding, np.array(node.embedding))
                scored.append((idx, sim))

            # Sort by similarity
            scored.sort(key=lambda x: x[1], reverse=True)

            # Take top branches
            top_branches = scored[:branch_factor]

            for idx, sim in top_branches:
                node = self.nodes[idx]

                if node.level == 0:
                    # Leaf node - collect it
                    collected_leaves.append({
                        "index": idx,
                        "text": node.text,
                        "level": node.level,
                        "source": node.source,
                        "similarity": sim,
                        "path_depth": depth
                    })
                else:
                    # Internal node - traverse children
                    if node.children:
                        traverse(node.children, depth + 1)
                    else:
                        # Summary node with no children (shouldn't happen but handle it)
                        collected_leaves.append({
                            "index": idx,
                            "text": node.text,
                            "level": node.level,
                            "source": node.source,
                            "similarity": sim,
                            "path_depth": depth
                        })

        # Start traversal from roots
        traverse(root_indices)

        # Sort collected leaves by similarity and return top_k
        collected_leaves.sort(key=lambda x: x["similarity"], reverse=True)
        return collected_leaves[:top_k]

    def collapsed_tree_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Collapsed tree retrieval - includes both summaries AND leaves.

        This gives the LLM both high-level context (summaries) and
        specific details (leaves) for better answers.
        """
        query_embedding = np.array(self._get_embedding(query))

        # Score all nodes
        all_scored = []
        for i, node in enumerate(self.nodes):
            sim = self._cosine_similarity(query_embedding, np.array(node.embedding))
            all_scored.append({
                "index": i,
                "text": node.text,
                "level": node.level,
                "source": node.source,
                "similarity": sim
            })

        # Sort by similarity
        all_scored.sort(key=lambda x: x["similarity"], reverse=True)

        # Select diverse results: mix of levels
        selected = []
        level_counts = {}

        for item in all_scored:
            level = item["level"]
            level_counts[level] = level_counts.get(level, 0)

            # Allow up to 2 items per level, prioritize by score
            if level_counts[level] < 2 or len(selected) < top_k // 2:
                selected.append(item)
                level_counts[level] += 1

            if len(selected) >= top_k:
                break

        return selected[:top_k]


if __name__ == "__main__":
    txt_dir = Path(__file__).parent / "docs" / "txt"
    index_path = Path(__file__).parent / "raptor_index.json"

    doc_paths = list(txt_dir.glob("*.txt"))

    raptor = RaptorIndex()
    raptor.index_documents(doc_paths, max_levels=3)
    raptor.save(index_path)

    # Test search
    print("\n--- Test Search ---")
    results = raptor.search("What is the outlook for silver prices?")
    for r in results[:3]:
        print(f"\n[Level {r['level']}, Score: {r['similarity']:.3f}]")
        print(f"Source: {r['source']}")
        print(r['text'][:200] + "...")
