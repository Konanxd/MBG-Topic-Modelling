import re
import pandas as pd
import numpy as np
from typing import Optional, List
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndonesianTextPreprocessor:
    def __init__(self,
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = True,
                 remove_numbers: bool = True,
                 min_length: int = 3,
                 max_length: int = 1000,
                 use_stemming: bool = True,
                 use_stopword_removal: bool = True):
        """
        Initialize preprocessor with configuration.
        
        Args:
            remove_urls: Remove URLs (they don't help topic modeling)
            remove_mentions: Remove @mentions (usually not topical)
            remove_hashtags: Remove #hashtags (keep them - often topical)
            remove_numbers: Remove numbers (usually not meaningful for topics)
            min_length: Minimum text length (filter out too short tweets)
            max_length: Maximum text length (sanity check)
            use_stemming: Use Indonesian stemming (reduces word variations)
            use_stopword_removal: Remove Indonesian stopwords (common words)
        """
        self.remove_urls = remove_urls 
        self.remove_mentions = remove_mentions 
        self.remove_hashtags = remove_hashtags 
        self.remove_numbers = remove_numbers 
        self.min_length = min_length 
        self.max_length = max_length 
        self.use_stemming = use_stemming
        self.use_stopword_removal = use_stopword_removal

        if self.use_stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            logger.info("Stemmer initialized")

        if self.use_stopword_removal:
            factory = StopWordRemoverFactory()
            self.stopword_remover = factory.create_stop_word_remover()
            logger.info("Stopword remover initialized")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)

        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)

        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            text = re.sub(r'[^\w\s]', '', text)

        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        text = text.lower()

        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords_and_stem(self, text: str) -> str:
        if not text:
            return ""

        if self.use_stopword_removal:
            text = self.stopword_remover.remove(text)
        
        if self.use_stemming:
            text = self.stemmer.stem(text)
        
        return text

    def is_valid_text(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False

        length = len(text)
        return self.min_length <= length <= self.max_length
    
    def preprocess(self, texts: List[str], show_progress: bool = True) -> pd.DataFrame:
        logger.info(f"Starting preprocessing of {len(texts)} texts")

        results = []

        for idx, text in enumerate(texts):
            if show_progress and idx % 5000 == 0:
                logger.info(f"Processed {idx}/{len(texts)} texts")

            cleaned = self.clean_text(text)

            processed = self.remove_stopwords_and_stem(cleaned)

            is_valid = self.is_valid_text(text)

            results.append({
                'original_text': text,
                'cleaned_text': cleaned,
                'processed_text': processed,
                'is_valid': is_valid,
                'original_length': len(str(text)) if text else 0,
                'processed_length': len(str(processed)),
            })
        
        df = pd.DataFrame(results)

        valid_count = df['is_valid'].sum()
        filtered_count = len(df) - valid_count
        logger.info(f"Preprocessing complete: {valid_count} valid texts, {filtered_count} filtered out")

        return df
    
    def get_preprocessed_texts(self, df: pd.DataFrame) -> List[str]:
        return df[df['is_valid']]['processed_text'].tolist()