import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import streamlit as st

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# === TextSimilarityRecommender Class ===
class TextSimilarityRecommender:
    def __init__(self, dataset_path):
        """Initialize the recommender with a dataset and build the necessary components."""
        self.df = pd.read_csv(dataset_path)
        
        # Fallback if Category column is missing
        if 'Category' not in self.df.columns:
            self.df['Category'] = 'Unknown'

        self.df = self.df.dropna(subset=['Topic_Text_B'])  # Remove rows with missing documents
        self.documents = self.df['Topic_Text_B'].astype(str).tolist()
        self.titles = self.df['Topic_Text_A'].astype(str).tolist()
        self.categories = self.df['Category'].astype(str).tolist()
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.processed_docs = self._preprocess_documents(self.documents)
        self.vectorizer, self.tfidf_matrix = self._build_tfidf_matrix(self.processed_docs)
        self.terms = self.vectorizer.get_feature_names_out()
        self.inverted_index = self._create_inverted_index(self.tfidf_matrix, self.terms)

    def _preprocess(self, text):
        """Preprocess the text by lowercasing, removing punctuation/numbers, tokenizing, and lemmatizing."""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation + string.digits)}]', ' ', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def _preprocess_documents(self, documents):
        """Preprocess all documents."""
        return [self._preprocess(doc) for doc in documents]

    def _build_tfidf_matrix(self, processed_docs):
        """Build TF-IDF matrix from processed documents."""
        vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english', ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        return vectorizer, tfidf_matrix

    def _create_inverted_index(self, tfidf_matrix, terms):
        """Create an inverted index for fast document retrieval."""
        inverted_index = defaultdict(set)
        for term_index, term in enumerate(terms):
            for doc_index in tfidf_matrix[:, term_index].nonzero()[0]:
                inverted_index[term].add(doc_index)
        return inverted_index

    def find_similar_texts(self, idx, top_k=10):
        """Find similar texts for a given document index."""
        query_vector = self.tfidf_matrix[idx]
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        return top_indices, similarity_scores[top_indices]

    def display_results(self, idx, similar_indices, similarity_scores):
        """Display the results in a formatted way."""
        st.write(f"\n🔍 Top Articles for Document Index {idx}:")
        st.write(f"**Title**: {self.titles[idx][:100]}")
        st.write(f"**Snippet**: {self.documents[idx][:150]}...")
        st.write("\n🔎 Similar Documents:")
        for rank, (i, score) in enumerate(zip(similar_indices, similarity_scores), 1):
            st.write(f"\n📘 Result #{rank}")
            st.write(f"**Title**: {self.titles[i][:100]}")
            st.write(f"**Similarity Score**: {score:.4f}")
            st.write(f"**Snippet**: {self.documents[i][:150]}...")

    def plot_similarity_scores(self, similarity_scores, top_k):
        """Plot the similarity scores for the top K recommendations."""
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(1, top_k + 1), similarity_scores, color='#6A5ACD', edgecolor='black')
        ax.set_xticks(range(1, top_k + 1))
        ax.set_xticklabels([f"#{i}" for i in range(1, top_k + 1)], fontsize=10)
        ax.set_xlabel('Recommendation Rank', fontsize=12)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title(f"Top {top_k} Articles Similarity Scores", fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)

    def export_to_csv(self, similar_indices, similarity_scores, top_k, idx):
        """Export the top K results to a CSV file."""
        export_data = []
        for rank, (i, score) in enumerate(zip(similar_indices, similarity_scores), 1):
            export_data.append({
                'Rank': rank,
                'Title': self.titles[i],
                'Similarity Score': round(score, 4),
                'Snippet': self.documents[i][:300],
                'Category': self.categories[i] if i < len(self.categories) else "Unknown",
            })

        export_df = pd.DataFrame(export_data)
        filename = f'recommended_articles_{idx}.csv'
        export_df.to_csv(filename, index=False)
        st.download_button("Download CSV", filename=filename)

# === Streamlit Interface ===
def main():
    # Load the recommender
    recommender = TextSimilarityRecommender('Text_Similarity_Dataset_with_Categories.csv')

    # Streamlit UI
    st.title("Text Similarity Recommender")
    st.sidebar.header("Options")
    
    option = st.sidebar.radio("Choose an option", ["Find Similar Texts", "Show All Document Indices", "Exit"])

    if option == "Find Similar Texts":
        idx = st.number_input("Enter document index:", min_value=0, max_value=len(recommender.documents)-1, value=0)
        
        if st.button("Find Similar"):
            similar_indices, similar_scores = recommender.find_similar_texts(idx)
            recommender.display_results(idx, similar_indices, similar_scores)
            recommender.plot_similarity_scores(similar_scores, len(similar_indices))

            if st.checkbox("Export to CSV"):
                recommender.export_to_csv(similar_indices, similar_scores, len(similar_indices), idx)

    elif option == "Show All Document Indices":
        st.write("\nDocument Index : Category : Content Preview")
        st.write("-" * 60)
        for i, (cat, doc) in enumerate(zip(recommender.categories, recommender.documents)):
            st.write(f"{i} : {cat} : {doc[:50]}...")

    elif option == "Exit":
        st.write("Thank you for using the Text Similarity Recommender!")

if __name__ == "__main__":
    main()
