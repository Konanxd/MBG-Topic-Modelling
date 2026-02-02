import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicVisualizer:
    def __init__(self, model, output_dir: str = "result/figures/"):
        self.model = model
        self.output_dir = output_dir
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        logger.info(f"Visualizer initialized. Outputs will be saved to {output_dir}")

    def plot_topic_distribution(self,
                                top_n: int = 20,
                                save: bool = True,
                                filename: str = "topic_distribution.png") -> None:
        logger.info("Plotting topic distribution")

        topic_info = self.model.get_topic_info()

        topic_info = topic_info[topic_info['Topic'] != -1]
        topic_info = topic_info.head(top_n)

        fig, ax = plt.subplots(figsize=(14, 8))

        bars = ax.barh(
            range(len(topic_info)),
            topic_info['Count'],
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(topic_info)))
        )

        topic_labels = [
            f"Topic {row['Topic']}: {row['Name'].split('_')[1:4]}"[:50]
            for _, row in topic_info.itterows()
        ]

        ax.set_yticks(range(len(topic_info)))
        ax.set_yticklabels(topic_labels)

        ax.set_xlabel('Number of Documents', fontsize=12, fontweight='bold')
        ax.set_ylabel('Topic', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Topics by Document Count', fontsize=14, fontweight='bold', pad=20)

        for i, (bar, count) in enumerate(zip(bars, topic_info['Count'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_()/2,
                    f' {count}',
                    ha='left', va='center', fontsize=9)

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.output_dir}{filename}", dpi=300, bbox_inches='tight')

        plt.show()

    def plot_topic_wordcloud(self,
                             topic_id: int,
                             save: bool = True,
                             filename: Optional[str] = None) -> None:
        logger.info(f"Generating word cloud for topic {topic_id}")

        topic_words = self.model.get_topic_words(topic_id, top_n=50)

        if not topic_words:
            logger.warning(f"No words found for topic {topic_id}")
            
            return

        word_freq = {word: score for word, score in topic_words}

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Topic {topic_id} - Word Cloud', 
                    fontsize=14, fontweight='bold', pad=20)
        
        top_5_words = ', '.join([word for word, _ in topic_words[:5]])
        ax.text(0.5, -0.05, f"Top words: {top_5_words}",
               transform=ax.transAxes,
               ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"wordcloud_topic_{topic_id}.png"
            plt.savefig(f"{self.output_dir}{filename}", dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {self.output_dir}{filename}")
        
        plt.show()
    
    def plot_multiple_wordclouds(self, 
                                 topic_ids: List[int],
                                 save: bool = True,
                                 filename: str = "wordclouds_grid.png") -> None:
        logger.info(f"Generating word cloud grid for {len(topic_ids)} topics")

        n_topics = len(topic_ids)
        n_cols = 3
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_topics > 1 else [axes]

        for idx, topic_id in enumerate(topic_ids):
            topic_words = self.model.get_topic_words(topic_id, top_n=30)
            
            if not topic_words:
                axes[idx].axis('off')
                continue
            
            word_freq = {word: score for word, score in topic_words}
            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap='viridis',
                relative_scaling=0.5
            ).generate_from_frequencies(word_freq)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            
            top_3_words = ', '.join([word for word, _ in topic_words[:3]])
            axes[idx].set_title(f'Topic {topic_id}\n{top_3_words}', 
                               fontsize=10, fontweight='bold')
        
        for idx in range(len(topic_ids), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}{filename}", dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {self.output_dir}{filename}")
        
        plt.show()

    def plot_topic_hierarchy(self, 
                            save: bool = True,
                            filename: str = "topic_hierarchy.html") -> None:
        logger.info("Generating topic hierarchy visualization")
        
        try:
            fig = self.model.model.visualize_hierarchy()
            
            if save:
                fig.write_html(f"{self.output_dir}{filename}")
                logger.info(f"Saved to {self.output_dir}{filename}")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error generating hierarchy: {str(e)}")
            logger.info("Hierarchy requires sufficient topics. Try with more topics.")

    def plot_topics_over_time(self,
                             documents: List[str],
                             timestamps: List[str],
                             save: bool = True,
                             filename: str = "topics_over_time.html") -> None:
        logger.info("Generating topics over time visualization")
        
        try:
            timestamps = pd.to_datetime(timestamps)
            
            topics_over_time = self.model.model.topics_over_time(
                documents, 
                timestamps,
                nr_bins=20  # Number of time periods
            )
            
            fig = self.model.model.visualize_topics_over_time(topics_over_time)
            
            if save:
                fig.write_html(f"{self.output_dir}{filename}")
                logger.info(f"Saved to {self.output_dir}{filename}")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error generating temporal analysis: {str(e)}")

    def plot_topic_similarity_heatmap(self,
                                     top_n: int = 20,
                                     save: bool = True,
                                     filename: str = "topic_similarity.png") -> None:        
        logger.info("Generating topic similarity heatmap")
        
        try:
            topic_info = self.model.get_topic_info()
            topic_info = topic_info[topic_info['Topic'] != -1].head(top_n)
            topic_ids = topic_info['Topic'].tolist()
            
            c_tf_idf = self.model.model.c_tf_idf_
            
            topic_embeddings = c_tf_idf[topic_ids].toarray()
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(topic_embeddings)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            sns.heatmap(
                similarity_matrix,
                annot=False,  # Too cluttered with annotations
                cmap='RdYlBu_r',  # Red=similar, Blue=different
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Cosine Similarity'},
                vmin=0, vmax=1,
                ax=ax
            )
            
            topic_labels = [f"T{tid}" for tid in topic_ids]
            ax.set_xticklabels(topic_labels, rotation=45, ha='right')
            ax.set_yticklabels(topic_labels, rotation=0)
            
            ax.set_title('Topic Similarity Matrix', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(f"{self.output_dir}{filename}", dpi=300, bbox_inches='tight')
                logger.info(f"Saved to {self.output_dir}{filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error generating similarity heatmap: {str(e)}")

    def plot_document_topic_distribution(self,
                                        documents: List[str],
                                        sample_size: int = 1000,
                                        save: bool = True,
                                        filename: str = "doc_topic_dist.png") -> None:
        logger.info("Generating document-topic distribution")
        
        if self.model.probs is None:
            logger.warning("Probabilities not calculated. Enable calculate_probabilities=True")
            return
        
        if len(documents) > sample_size:
            indices = np.random.choice(len(documents), sample_size, replace=False)
            probs = self.model.probs[indices]
        else:
            probs = self.model.probs
        
        max_probs = np.max(probs, axis=1) if probs.ndim > 1 else probs
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(max_probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Maximum Topic Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Document-Topic Assignment Confidence', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, 
                  label='Low confidence threshold')
        ax.axvline(0.8, color='green', linestyle='--', alpha=0.5,
                  label='High confidence threshold')
        
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}{filename}", dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {self.output_dir}{filename}")
        
        plt.show()
    
    def generate_topic_report(self, 
                            output_file: str = "topic_report.txt") -> None:
        logger.info("Generating topic report")
        
        topic_info = self.model.get_topic_info()
        
        with open(f"{self.output_dir}{output_file}", 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TOPIC MODELING REPORT\n")
            f.write("="*80 + "\n\n")
            
            n_topics = len(topic_info[topic_info['Topic'] != -1])
            n_outliers = topic_info[topic_info['Topic'] == -1]['Count'].sum()
            total_docs = topic_info['Count'].sum()
            
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Topics: {n_topics}\n")
            f.write(f"Total Documents: {total_docs}\n")
            f.write(f"Outlier Documents: {n_outliers} ({n_outliers/total_docs*100:.2f}%)\n")
            f.write(f"Average Documents per Topic: {total_docs/n_topics:.0f}\n\n")
            
            f.write("TOPICS\n")
            f.write("="*80 + "\n\n")
            
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                
                if topic_id == -1:
                    f.write("Topic -1: OUTLIERS\n")
                    f.write(f"  Documents: {row['Count']}\n\n")
                    continue
                
                f.write(f"Topic {topic_id}\n")
                f.write("-"*80 + "\n")
                f.write(f"  Documents: {row['Count']}\n")
                f.write(f"  Name: {row['Name']}\n")
                
                topic_words = self.model.get_topic_words(topic_id, top_n=15)
                f.write("  Top Words:\n")
                for word, score in topic_words:
                    f.write(f"    - {word}: {score:.4f}\n")
                
                f.write("\n")
        
        logger.info(f"Report saved to {self.output_dir}{output_file}")