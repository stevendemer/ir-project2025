import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

class SearchEngine:
    
    def __init__(self, data_path):
        print('--- Initializing search engine ---')
        self.vectorizer = None
        self.tfidf_matrix = None
        self.df = None

        self._load_data(data_path)
        self._build_index()
    
    def _load_data(self, path: str):
        print(f'Loading data from {path}...')
        try:
            self.df = pd.read_parquet(path)
        except Exception:
            # fallback - in case it was saved as csv
            self.df = pd.read_csv(path.replace('.parquet', '.csv'))
    
        self.df['processed_speech'] = self.df['processed_speech'].fillna('')
        print(f'Loaded {len(self.df)} speeches.')
    
    
    def _build_index(self):
        print("Building TF-IDF index... (This might take memory)")
        start = time.time()

        # keep the 10000 most important words
        self.vectorizer = TfidfVectorizer(max_features=10000, strip_accents='unicode')

        # from text to matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_speech'])
        
        print(f'Index built in {time.time() - start:.2f} seconds')
    
    
    def search(self, query:str, top_k=5):

        # preprocess the query so it matches our sample data
        # space lemmatizer - in the future
        query_vec = self.vectorizer.transform([query.lower()])

        # cosine similarity
        # compare the query vec with all the speeches
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Κρατάει τα τελευταία k στοιχεία (που είναι αυτά με τα μεγαλύτερα σκορ)
        # αντιστρέφει ώστε το πρώτο να είναι αυτό με το μεγαλύτερο σκορ (φθίνουσα σειρά).
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.0:
                record = self.df.iloc[idx].to_dict()
                record['similarity_score'] = round(float(score), 4)
                results.append(record)

        return results
                