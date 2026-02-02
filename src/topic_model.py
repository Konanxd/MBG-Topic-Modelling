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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeler:
    def __init__(self,
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 nr_topics: Optional[int] = None,
                 min_topic_size: int = 30,
                 n_gram_range: Tuple[int, int] = (1,2),
                 calculate_probabilities: bool = True):
        self.embedding_model = embedding_model
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.n_gram_range = n_gram_range
        self.calculate_probabilities = calculate_probabilities

        self.model = None
        self.embeddings = None
        self.topics = None
        self.probs = None

        logger.info(f"TopicModeler initialized with model: {self.embedding_model}")

    def _create_embedding_model(self) -> SentenceTransformer:
        logger.info(f"Loading embedding model: {self.embedding_model}")
        model = SentenceTransformer(self.embedding_model)
        return model

    def _create_umap_model(self) -> UMAP:
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
        )

        return umap_model

    def _create_hdbscan_model(self) -> HDBSCAN:
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
        ) 

        return hdbscan_model

    def _create_vectorizer_model(self) -> CountVectorizer:
        vectorizer_model = CountVectorizer(
            ngram_range=self.n_gram_range,
            min_df=5,
            max_df=0.8,
            stop_words='english',   
        )          

        return vectorizer_model 

    def train(self, documents: List[str]) -> 'TopicModeler':
        logger.info(f"Starting training to {len(documents)} documents")

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

        logger.info("Fitting BERTopic model...")
        self.topics, self.probs = self.model.fit_transform(documents)

        self.embeddings = self.model._extract_embeddings(
            documents,
            method="document",
            verbose=True
        )

        num_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        outliers = sum(1 for t in self.topics if t == -1)
        logger.info(f"Training complete!")
        logger.info(f"Found {num_topics} topics")
        logger.info(f"Outliers: {outliers} documents ({outliers/len(documents)*100:.2f}%)")

        return self

    def get_topic_info(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.get_topic_info()
    
    def get_topic_words(self, topic_id: int, topic_n: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.get_topic(topic_id, topic_n)
    
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
        self.model.reduce_topics(self.topics, nr_topicss=nr_topics)

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
            with open(embedding_path, 'wb') as f:
                self.embeddings = pickle.load(f)
        except FileNotFoundError:
            logger.warning("Embedding file not found. Visualization may be limited.")

        logger.info("Model loaded successfully")
        return self
    