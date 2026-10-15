"""
App2Vec — Graph Embedding Module via Node2Vec
Learns dense vector representations of apps from their transition patterns.

Architecture:
  1. Build a directed, weighted App Transition Graph from event data
  2. Run biased random walks (Node2Vec) over the graph
  3. Train Word2Vec (Skip-gram) on walk sequences to learn embeddings
  4. Apply PCA compression: 64-dim -> 8-dim per app (prevent overfitting)

Output: Each app gets an 8-dimensional embedding vector.
        Semantically similar apps (Instagram <-> TikTok) cluster together.
        Dissimilar apps (Settings <-> BGMI) are far apart.

Reference: Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks",
           KDD 2016. https://arxiv.org/abs/1607.00653
"""
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


class App2Vec:
    """
    Learns app embeddings from transition graphs using Node2Vec-style
    biased random walks + Word2Vec skip-gram training.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        pca_dim: int = 8,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,
        q: float = 0.5,
        window: int = 5,
        epochs: int = 50,
        seed: int = 42,
    ):
        """
        Args:
            embedding_dim: Raw Word2Vec embedding dimension (before PCA).
            pca_dim: Final compressed dimension per app after PCA.
            walk_length: Length of each random walk.
            num_walks: Number of walks per node.
            p: Return parameter (1.0 = balanced BFS/DFS).
            q: In-out parameter (<1.0 = biased toward local neighborhood).
            window: Word2Vec context window size.
            epochs: Word2Vec training epochs.
            seed: Random seed for reproducibility.
        """
        self.embedding_dim = embedding_dim
        self.pca_dim = pca_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.epochs = epochs
        self.seed = seed

        self.graph: Optional[nx.DiGraph] = None
        self.w2v_model: Optional[Word2Vec] = None
        self.pca: Optional[PCA] = None
        self.app_names: List[str] = []
        self.raw_embeddings: Dict[str, np.ndarray] = {}
        self.compressed_embeddings: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed, weighted transition graph from event data.
        
        Args:
            df: DataFrame with 'current_app' and 'next_app' columns.
            
        Returns:
            Directed graph with edge weights = transition counts.
        """
        print("    Building App Transition Graph...")
        
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            src = str(row["current_app"])
            dst = str(row.get("next_app", ""))
            if src and dst and src != dst:
                transition_counts[src][dst] += 1

        # Build NetworkX directed graph
        G = nx.DiGraph()
        for src, targets in transition_counts.items():
            for dst, count in targets.items():
                G.add_edge(src, dst, weight=count)

        self.graph = G
        self.app_names = sorted(G.nodes())

        print(f"      Nodes (apps): {G.number_of_nodes()}")
        print(f"      Edges (transitions): {G.number_of_edges()}")
        print(f"      Total transition weight: {sum(d['weight'] for _, _, d in G.edges(data=True)):,}")

        return G

    def _node2vec_walk(self, start_node: str, rng: np.random.RandomState) -> List[str]:
        """
        Perform a single biased random walk starting from start_node.
        Uses Node2Vec's p, q parameters for walk bias.
        """
        walk = [start_node]
        
        for _ in range(self.walk_length - 1):
            cur = walk[-1]
            neighbors = list(self.graph.successors(cur))
            
            if not neighbors:
                break
            
            if len(walk) < 2:
                # First step: weighted random choice
                weights = [self.graph[cur][n]["weight"] for n in neighbors]
                total = sum(weights)
                probs = [w / total for w in weights]
                next_node = rng.choice(neighbors, p=probs)
            else:
                # Biased walk using p (return) and q (in-out) parameters
                prev = walk[-2]
                weights = []
                for n in neighbors:
                    edge_weight = self.graph[cur][n]["weight"]
                    if n == prev:
                        # Return to previous node: weight / p
                        weights.append(edge_weight / self.p)
                    elif self.graph.has_edge(prev, n):
                        # Neighbor of both prev and cur (BFS-like): weight * 1
                        weights.append(edge_weight)
                    else:
                        # Exploration (DFS-like): weight / q
                        weights.append(edge_weight / self.q)
                
                total = sum(weights)
                if total == 0:
                    break
                probs = [w / total for w in weights]
                next_node = rng.choice(neighbors, p=probs)
            
            walk.append(next_node)
        
        return walk

    def _generate_walks(self) -> List[List[str]]:
        """Generate all random walks for training."""
        print(f"    Generating {self.num_walks} walks/node x {len(self.app_names)} nodes "
              f"(walk_length={self.walk_length})...")
        
        rng = np.random.RandomState(self.seed)
        walks = []
        
        for walk_num in range(self.num_walks):
            # Shuffle node order each iteration for randomness
            nodes = list(self.app_names)
            rng.shuffle(nodes)
            for node in nodes:
                walk = self._node2vec_walk(node, rng)
                walks.append(walk)
        
        print(f"      Generated {len(walks)} walks")
        return walks

    def train(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Full training pipeline: build graph -> walk -> Word2Vec -> PCA.
        
        Args:
            df: DataFrame with 'current_app' and 'next_app' columns.
            
        Returns:
            Dictionary mapping app_name -> compressed embedding (pca_dim,).
        """
        # Step 1: Build transition graph
        self.build_graph(df)
        
        if self.graph.number_of_nodes() < 2:
            raise ValueError("Need at least 2 apps to build embeddings.")
        
        # Step 2: Generate random walks
        walks = self._generate_walks()
        
        # Step 3: Train Word2Vec (Skip-gram)
        print(f"    Training Word2Vec (dim={self.embedding_dim}, window={self.window}, "
              f"epochs={self.epochs})...")
        
        self.w2v_model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=1,
            sg=1,  # Skip-gram
            workers=4,
            epochs=self.epochs,
            seed=self.seed,
        )
        
        # Extract raw embeddings
        self.raw_embeddings = {}
        for app in self.app_names:
            if app in self.w2v_model.wv:
                self.raw_embeddings[app] = self.w2v_model.wv[app]
            else:
                # Fallback: zero vector for unseen apps
                self.raw_embeddings[app] = np.zeros(self.embedding_dim, dtype=np.float32)
        
        print(f"      Learned embeddings for {len(self.raw_embeddings)} apps")
        
        # Step 4: PCA compression
        print(f"    Compressing {self.embedding_dim}-dim -> {self.pca_dim}-dim via PCA...")
        
        embedding_matrix = np.array([self.raw_embeddings[app] for app in self.app_names])
        
        # Adjust PCA dim if we have fewer apps than pca_dim
        actual_pca_dim = min(self.pca_dim, len(self.app_names), self.embedding_dim)
        
        self.pca = PCA(n_components=actual_pca_dim, random_state=self.seed)
        compressed_matrix = self.pca.fit_transform(embedding_matrix)
        
        variance_retained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"      PCA variance retained: {variance_retained:.1f}%")
        
        self.compressed_embeddings = {}
        for i, app in enumerate(self.app_names):
            self.compressed_embeddings[app] = compressed_matrix[i].astype(np.float32)
        
        self.is_fitted = True
        
        # Print similarity examples
        self._print_similarities()
        
        return self.compressed_embeddings

    def get_embedding(self, app_name: str, compressed: bool = True) -> np.ndarray:
        """
        Get embedding vector for an app.
        
        Args:
            app_name: Name of the app.
            compressed: If True, return PCA-compressed embedding.
            
        Returns:
            Embedding vector (pca_dim,) or (embedding_dim,).
        """
        if compressed:
            if app_name in self.compressed_embeddings:
                return self.compressed_embeddings[app_name]
            # Unknown app: return zero vector
            dim = self.pca_dim if self.pca else self.embedding_dim
            return np.zeros(dim, dtype=np.float32)
        else:
            if app_name in self.raw_embeddings:
                return self.raw_embeddings[app_name]
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def cosine_similarity(self, app1: str, app2: str) -> float:
        """Compute cosine similarity between two app embeddings."""
        v1 = self.get_embedding(app1, compressed=False)
        v2 = self.get_embedding(app2, compressed=False)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def most_similar(self, app_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar apps by cosine similarity."""
        if not self.is_fitted or app_name not in self.raw_embeddings:
            return []
        
        similarities = []
        for other in self.app_names:
            if other != app_name:
                sim = self.cosine_similarity(app_name, other)
                similarities.append((other, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _print_similarities(self):
        """Print example similarity comparisons for verification."""
        print(f"\n    Embedding Similarity Examples:")
        
        test_pairs = [
            ("Instagram", "TikTok", "Should be HIGH (both social/video)"),
            ("Instagram", "WhatsApp", "Should be MODERATE (social + messaging)"),
            ("WhatsApp", "Telegram", "Should be HIGH (both messaging)"),
            ("Gmail", "Calendar", "Should be HIGH (both productivity)"),
            ("BGMI", "Settings", "Should be LOW (gaming vs system)"),
            ("YouTube", "Netflix", "Should be HIGH (both entertainment)"),
            ("ChatGPT", "Docs", "Should be HIGH (both productivity/AI)"),
        ]
        
        for app1, app2, expectation in test_pairs:
            if app1 in self.raw_embeddings and app2 in self.raw_embeddings:
                sim = self.cosine_similarity(app1, app2)
                print(f"      cos({app1}, {app2}) = {sim:.3f}  ({expectation})")

    def get_transition_stats(self) -> pd.DataFrame:
        """Get transition graph statistics as a DataFrame."""
        if self.graph is None:
            return pd.DataFrame()
        
        rows = []
        for node in self.app_names:
            in_deg = self.graph.in_degree(node, weight="weight")
            out_deg = self.graph.out_degree(node, weight="weight")
            rows.append({
                "app": node,
                "in_transitions": in_deg,
                "out_transitions": out_deg,
                "total_transitions": in_deg + out_deg,
                "unique_successors": len(list(self.graph.successors(node))),
                "unique_predecessors": len(list(self.graph.predecessors(node))),
            })
        
        return pd.DataFrame(rows).sort_values("total_transitions", ascending=False)

    def save(self, dir_path: str) -> None:
        """Save all artifacts to directory."""
        os.makedirs(dir_path, exist_ok=True)
        
        # Save compressed embeddings as numpy
        embedding_matrix = np.array([self.compressed_embeddings[app] for app in self.app_names])
        np.save(os.path.join(dir_path, "app_embeddings.npy"), embedding_matrix)
        
        # Save app name ordering
        with open(os.path.join(dir_path, "app_names.json"), "w") as f:
            json.dump(self.app_names, f)
        
        # Save PCA
        with open(os.path.join(dir_path, "app2vec_pca.pkl"), "wb") as f:
            pickle.dump(self.pca, f)
        
        # Save raw embeddings dict
        raw_dict = {k: v.tolist() for k, v in self.raw_embeddings.items()}
        with open(os.path.join(dir_path, "app_raw_embeddings.json"), "w") as f:
            json.dump(raw_dict, f, indent=2)
        
        # Save compressed embeddings dict
        comp_dict = {k: v.tolist() for k, v in self.compressed_embeddings.items()}
        with open(os.path.join(dir_path, "app_compressed_embeddings.json"), "w") as f:
            json.dump(comp_dict, f, indent=2)
        
        print(f"    Saved App2Vec artifacts to {dir_path}/")

    @classmethod
    def load(cls, dir_path: str) -> "App2Vec":
        """Load pre-trained App2Vec from directory."""
        instance = cls()
        
        with open(os.path.join(dir_path, "app_names.json"), "r") as f:
            instance.app_names = json.load(f)
        
        with open(os.path.join(dir_path, "app2vec_pca.pkl"), "rb") as f:
            instance.pca = pickle.load(f)
        
        embedding_matrix = np.load(os.path.join(dir_path, "app_embeddings.npy"))
        instance.compressed_embeddings = {
            app: embedding_matrix[i] for i, app in enumerate(instance.app_names)
        }
        
        # Load raw embeddings if available
        raw_path = os.path.join(dir_path, "app_raw_embeddings.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r") as f:
                raw_dict = json.load(f)
            instance.raw_embeddings = {k: np.array(v, dtype=np.float32) for k, v in raw_dict.items()}
        
        instance.is_fitted = True
        return instance
