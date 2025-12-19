import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import spacy
import os


try:
    nlp = spacy.load("el_core_news_sm")
except OSError:
    print("el_core_news_sm not found. Please install it with python -m spacy download el_core_news_sm")
    exit()

class AnalyticsEngine:
    def __init__(self, data_path:str):
        
        if data_path.endswith('.parquet'):
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        
        self.df['sitting_date'] = pd.to_datetime(self.df['sitting_date'], format='%d/%m/%Y', errors='coerce')
        self.df['year'] = self.df['sitting_date'].dt.year
    
    def get_keywords_by_group(self, group_col, top_n=10):
        """
        Keywords per party or .
        """

        # join texts per group
        grouped_df = self.df.groupby(group_col)['processed_speech'].apply(lambda x: " ".join(x)).reset_index()

        # TF - IDF
        tfidf = TfidfVectorizer(max_features=1000, max_df=0.8)
        try:
            tfidf_matrix = tfidf.fit_transform(grouped_df['processed_speech'])
        except ValueError:
            return {}
        
        feature_names = tfidf.get_feature_names_out()
        
        results = {}
        
        for i, row in grouped_df.iterrows():
            group_name = row[group_col]

            # convert numpy types to native python ones for clean output
            if pd.isna(group_name): continue
            
            row_vector = tfidf_matrix[i].toarray().flatten()

            top_indices = row_vector.argsort()[-top_n:][::-1]
            
            keywords = [(feature_names[idx], round(row_vector[idx], 3)) for idx in top_indices]
            results[group_name] = keywords
        
        return results

    
    def get_keywords_timeline(self, keyword:str):
        """
        Timeseries for keyword
        """
        
        yearly_counts = self.df[self.df['processed_speech'].str.contains(keyword, na=False)].groupby('year').size()
        total_speeches = self.df.groupby('year').size()
        
        timeline = (yearly_counts / total_speeches).fillna(0) * 100
        
        return timeline.sort_index()
    
    # task 3
    def get_top_similar_pairs(self, top_k=20):
        """
        Pairwise similarity between MPs
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # filter for active members only to reduce matrix size
        counts = self.df['member_name'].value_counts()
        active_members = counts[counts >= 10].index # MPs with at least 10 active speeches
        subset = self.df[self.df['member_name'].isin(active_members)]
        
        # Group text by member
        grouped = subset.groupby('member_name')['processed_speech'].apply(lambda x: " ".join(x)).reset_index()
        member_names = grouped['member_name'].tolist()

        # vectorization (tf-idf)
        tfidf = TfidfVectorizer(max_features=2000)
        tfidf_matrix = tfidf.fit_transform(grouped['processed_speech'])

        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # zero diagonal 
        np.fill_diagonal(similarity_matrix, 0)
        
        pairs = []
        
        rows, cols = similarity_matrix.shape

        # get top indices flat
        flat_indices = np.argsort(similarity_matrix.ravel())[-top_k*2:]

        seen_pairs = set()
        
        for idx in flat_indices:
            i, j = np.unravel_index(idx, (rows, cols))
            if i < j: # upper triangle only
                score = similarity_matrix[i, j]
                pair_key = tuple(sorted((member_names[i], member_names[j])))

                if score > 0.01 and pair_key not in seen_pairs: 
                    seen_pairs.add(pair_key)
                    pairs.append({
                        "Member A": member_names[i],
                        "Member B": member_names[j],
                        "Similarity": round(float(score), 4)
                    })
        
        pairs = sorted(pairs, key=lambda x: x['Similarity'], reverse=True)[:top_k]
        return pairs
                    
    def perform_lsi(self, n_topics=5):
        """
        LSI for top modeling.
        """
        tfidf = TfidfVectorizer(max_features=5000, max_df=0.5, min_df=5, strip_accents='unicode')
        
        # speeches with actual content
        valid_speeches = self.df[self.df['processed_speech'].str.len() > 10]['processed_speech']
        tfidf_matrix = tfidf.fit_transform(valid_speeches)
        feature_names = tfidf.get_feature_names_out()
        
        svd = TruncatedSVD(n_components=n_topics, random_state=42)
        svd.fit(tfidf_matrix)

        topics = {}

        for index, topic in enumerate(svd.components_):
            top_words_indices = topic.argsort()[-10:][::-1]
            top_words = [(feature_names[i], round(topic[i], 3)) for i in top_words_indices]
            topics[f'Topic {index + 1}'] = top_words

        return topics
    
    
    def get_sentiment_by_party(self):
        """
        Custom task: sentiment analysis using Spacy Lemmatization.
        Calculates the average sentiment score per party based on a dictionary of words.
        """
        positive_lemmas = {
            'ανάπτυξη', 'πρόοδος', 'επιτυχία', 'θετικός', 'λύση', 'μέλλον', 'ελπίδα', 
            'έργο', 'σταθερότητα', 'αύξηση', 'κέρδος', 'δικαιοσύνη', 'ασφάλεια', 
            'δημιουργία', 'ευημερία', 'εμπιστοσύνη', 'στήριξη', 'ενίσχυση'
        }
        
        negative_lemmas = {
            'κρίση', 'πρόβλημα', 'αποτυχία', 'λάθος', 'καταστροφή', 'χρέος', 'ύφεση', 
            'ανεργία', 'μείωση', 'φτώχεια', 'βία', 'σκάνδαλο', 'διαφθορά', 'κίνδυνος', 
            'απειλή', 'ανικανότητα', 'ψέμα', 'ντροπή'
        }

        def sentiment_scorer(text):
            doc = nlp(text, disable=["ner", "parser"])
            
            score = 0
            count = 0

            for token in doc:
                lemma = token.lemma_.lower()
                
                if lemma in positive_lemmas:
                    score += 1
                elif lemma in negative_lemmas:
                    score -= 1
                
                if not token.is_stop and not token.is_punct:
                    count += 1
            
            # normalize
            return (score / count) if count > 0 else 0
        
        party_counts = self.df['political_party'].value_counts()
        major_parties = party_counts[party_counts > 10].index  # parties with more than 10 members
        subset = self.df[self.df['political_party'].isin(major_parties)].copy()
        
        print("Calculating sentiment with Spacy (this may take a while)...")
        subset['sentiment_score'] = subset['processed_speech'].apply(sentiment_scorer)
        
        return subset.groupby('political_party')['sentiment_score'].mean().sort_values(ascending=False)
        
        


   


       

