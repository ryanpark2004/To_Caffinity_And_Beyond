from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cossim(query, products_df):
    vectorizer = TfidfVectorizer()
    # tfidf of reviews
    tfidf_matrix = vectorizer.fit_transform(products_df['reviews'])
    # vec of query
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf)[0]
    return products_df.iloc[sims.argsort()[-5:][::-1]]


#run to compare wiht calebs'
def svd_recommend(query,ingredient, products_df, k=100, top_k=5, similarity_penalty=0.5):
    if ingredient:
        products_df = products_df[~products_df['ingredients'].str.contains(ingredient, case=False)]
    """
    this svd decomposition creates patterns and based on the analysis of top terms in each dimension,
    some of the most common are:
    dim 0: coffee
    dim 1: energy drink
    dim 2: fruit flavors
    dim 3: sweetness
    dim 4: healthy
    dim 5: no sugar
    etc
    basically the query is projected on these k=100 dimensions and higher abs values means it aligns m
    more strongly
    + means it matches the dimesions 
    - means it opposes it
    we penalize and reward based on the user query
    """
    documents = (
        products_df['description'].fillna('').astype(str) + ' ' +
        products_df['bullet_points'].fillna('').astype(str) + ' ' +
        products_df['flavors'].fillna('').astype(str) + ' ' +
        products_df['reviews'].fillna('').astype(str) + ' ' +
        products_df['whole_title'].fillna('').astype(str)
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7,min_df=2,ngram_range=(1, 2),analyzer='word')
    td_matrix = vectorizer.fit_transform(documents)
    svd = TruncatedSVD(n_components=k, random_state=42)
    doc_latent = svd.fit_transform(td_matrix)
    doc_latent = normalize(doc_latent)
    boosted_query= query.lower()
    if 'coffee' in boosted_query:
        boosted_query +=' brew beans roast'
    elif 'energy' in boosted_query:
        boosted_query += ' drink booster'
    elif 'juice' in boosted_query:
        boosted_query += ' fruit fresh'
    query_vec = vectorizer.transform([boosted_query])
    query_latent = normalize(query_vec.dot(svd.components_.T))
    #calc cossims
    sims = doc_latent.dot(query_latent.T).flatten()
    #based on your query, calc a score
    query_terms = set(query.lower().split())
    for i in range(len(sims)):
        product = products_df.iloc[i]
        product_text = (
            str(product['description']).lower() + ' ' +
            str(product['flavors']).lower() + ' ' +
            str(product['whole_title']).lower()
        )
        #calc the match
        match_score = sum(
            1 for term in query_terms 
            if term in product_text
        ) / len(query_terms) if query_terms else 0
        #if it's good match then boost it, otherwise strong penalty
        if match_score > 0.5:
            sims[i] *= (1 + match_score)
        else:
            sims[i] *= 0.3
    #remove the same brand or just products that seem very similar
    product_similarities = cosine_similarity(doc_latent)
    final_scores= sims.copy()
    top_N = min(20, len(products_df))
    candidate_indices= sims.argsort()[::-1][:top_N]
    for i in range(1, len(candidate_indices)):
        for j in range(i):
            if product_similarities[candidate_indices[i], candidate_indices[j]] > 0.8:
                final_scores[candidate_indices[i]]*= similarity_penalty
    top_indices = final_scores.argsort()[::-1][:top_k]
    return products_df.iloc[top_indices]

