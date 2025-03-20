import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cossim(query, products_df):
    vectorizer = TfidfVectorizer()
    # tfidf of reviews
    tfidf_matrix = vectorizer.fit_transform(products_df['reviews'])
    # vec of query
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    # indices of top 5
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    
    return products_df.iloc[top_indices]

