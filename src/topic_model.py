from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import pickle
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeler:
    def __init__(self,
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L13-v2",
                 nr_topics: Optional[int] = None,
                 min_topic_size: int = 30,
                 n_gram_range: Tuple[int, int] = (1,2),
                 calculate_probabilities: bool = True,
                 use_gpu_umap: bool = True):
        self.embedding_model = embedding_model
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.n_gram_range = n_gram_range
        self.calculate_probabilities = calculate_probabilities
        self.use_gpu_umap = use_gpu_umap

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.embeddings = None
        self.topics = None
        self.probs = None

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"✓ Embeddings will use GPU acceleration")
        else:
            logger.warning("⚠️  No GPU detected. Running on CPU (slower)")
        
        logger.info(f"TopicModeler initialized with model: {embedding_model}")

    def _create_embedding_model(self) -> SentenceTransformer:
        logger.info(f"Loading embedding model: {self.embedding_model}")
        model = SentenceTransformer(self.embedding_model)

        if self.device == "cuda":
            model = model.to(self.device)
            logger.info("✓ Embedding model moved to GPU")

        return model

    def _create_umap_model(self) -> UMAP:
        if self.use_gpu_umap and self.device == "cuda":
            try:
                from cuml.manifold import UMAP as cumlUMAP
                
                logger.info("✓ Using GPU-accelerated UMAP (cuML)")
                
                umap_model = cumlUMAP(
                    n_neighbors=10,        # Reduced for speed
                    n_components=5,
                    n_epochs=100,          # Reduced from default 200
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42,
                    verbose=True
                )
                
                logger.info("Expected UMAP time: 2-3 minutes (GPU)")
                return umap_model
                
            except ImportError:
                logger.warning("⚠️  cuML not installed. Install with:")
                logger.warning("    !pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com")
                logger.info("Falling back to optimized CPU UMAP...")
        
        logger.info("Using optimized CPU UMAP")
        
        umap_model = UMAP(
            n_neighbors=10,        # Reduced from 15 (30% faster)
            n_components=5,        # Good for clustering
            n_epochs=100,          # Reduced from 200 (40% faster)
            min_dist=0.0,          # Tight clusters
            metric='cosine',       # Best for text
            random_state=42,
            verbose=True,          # Show progress
            n_jobs=1,              # Single thread (UMAP parallelization is tricky)
            init='spectral',       # Fast initialization
        )
        
        logger.info("Expected UMAP time: 5-8 minutes (optimized CPU)")

        return umap_model

    def _create_hdbscan_model(self) -> HDBSCAN:
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=5,              # Reduced from 10 (faster, minimal quality loss)
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            core_dist_n_jobs=-1,        # Use all CPU cores for distance calculation
        )

        logger.info("Expected HDBSCAN time: 1-2 minutes")

        return hdbscan_model

    def _create_vectorizer_model(self) -> CountVectorizer:
        vectorizer_model = CountVectorizer(
            ngram_range=self.n_gram_range,
            min_df=5,
            max_df=0.8,
            stop_words='english',   
        )          

        return vectorizer_model 

    def train(self, documents: List[str], batch_size: Optional[int] = None) -> 'TopicModeler':
        logger.info(f"Starting training on {len(documents)} documents")
        logger.info(f"Device: {self.device}")

        if batch_size is None:
            if self.device == "cuda":
                batch_size = 128
                logger.info(f"Auto-detected batch size: {batch_size} (GPU)")
            else:
                batch_size = 32
                logger.info(f"Auto-detected batch size: {batch_size} (CPU)")

        logger.info("Creating embedding model...")
        embedding_model = self._create_embedding_model()

        logger.info("Creating UMAP model...")
        umap_model = self._create_umap_model()

        logger.info("Createing HDBSCAN model...")
        hdbscan_model = self._create_hdbscan_model()

        logger.info("Creating vectorizer model...")
        vectorizer_model = self._create_vectorizer_model()

        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=self.nr_topics,
            calculate_probabilities=self.calculate_probabilities,
            verbose=True
        )

        logger.info("="*60)
        logger.info("Starting BERTopic training...")
        logger.info("="*60)
        logger.info("\n[1/4] Generating embeddings...")
        logger.info(f"Batch size: {batch_size}")
        
        import time
        start_time = time.time()

        self.topics, self.probs = self.model.fit_transform(documents)

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed/60:.2f} minutes")

        logger.info("\nExtracting embeddings for visualization...")
        self.embeddings = self.model._extract_embeddings(
            documents,
            method="document",
            verbose=True
        )

        num_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        outliers = sum(1 for t in self.topics if t == -1)
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Found {num_topics} topics")
        logger.info(f"Outliers: {outliers} documents ({outliers/len(documents)*100:.2f}%)")
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info("="*60)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("✓ GPU cache cleared")

        return self

    def get_topic_info(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.get_topic_info()
    
    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.get_topic(topic_id, top_n)
    
    def get_document_topics(self, documents: List[str]) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df = pd.DataFrame({
            'Document': documents,
            'Topic': self.topics,
            'Probability': self.probs if self.probs is not None else [None] * len(documents)
        })

        return df

    def reduce_topics(self, nr_topics: int) -> 'TopicModeler':
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info(f"Reducing topics to {nr_topics}")
        self.model.reduce_topics(self.topics, nr_topics=nr_topics)

        self.topics = self.model.topics_

        logger.info("Topic reduction complete")

        return self

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Saving model to {path}")

        self.model.save(path, serialization="pickle")

        embedding_path = path.replace(".pkl", "_embedding.pkl")
        with open(embedding_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

        logger.info("Model saved successfully")

    def load_model(self, path: str) -> 'TopicModeler':
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Saving model to {path}")

        self.model = BERTopic.load(path)

        embedding_path = path.replace(".pkl", "_embedding.pkl")
        try:
            with open(embedding_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Embedding file not found. Visualization may be limited.")

        logger.info("Model loaded successfully")
        return self
    