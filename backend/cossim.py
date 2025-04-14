from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

# caleb's cossim:
def cossim(query, products_df):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(products_df['reviews'])
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf)[0]
    return products_df.iloc[sims.argsort()[-5:][::-1]]

#run to compare wiht calebs'
def svd_recommend(query, products_df, k=5, top_k=5):
    #in case it's missing, fill the missing strings 
    #"Great tasting coffee, Organic Fair trade Chocolate, Caramel Best coffee ever"
    documents = (
        products_df['description'].fillna('').astype(str) + ' ' +
        products_df['bullet_points'].fillna('').astype(str) + ' ' +
        products_df['flavors'].fillna('').astype(str) + ' ' +
        products_df['reviews'].fillna('').astype(str)
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=70)
    td_matrix = vectorizer.fit_transform(documents)
    #reduce the dimensions bro and process the query
    svd = TruncatedSVD(n_components=k)
    doc_latent = svd.fit_transform(td_matrix)
    doc_latent = normalize(doc_latent)
    query_vec = vectorizer.transform([query])
    query_latent = normalize(query_vec.dot(svd.components_.T))
    print()

    #calc the cossimalirities
    sims = doc_latent.dot(query_latent.T).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return products_df.iloc[top_indices]