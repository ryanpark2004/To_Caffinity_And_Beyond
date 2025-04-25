from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np


# caleb's cossim:
def cossim(query, products_df):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(products_df["reviews"])
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf)[0]
    return products_df.iloc[sims.argsort()[-5:][::-1]]


def print_svd_tags(vectorizer, svd, top_n=1):
    """
    prints the top tags per svd component:
    here is the top 10 tags for dims 0-11

    dim 0: energy drink | c4 | protein | free | zero | variety | brew | energy drinks | boost | cold
    dim 1: c4 | roast | brew | energy drink | cold brew | cold | protein | starbucks | vanilla | chocolate
    dim 2: red bull | bull | red | c4 | 24pk | oz 24pk | fl oz | fl | oz | 4pk
    dim 3: c4 | red bull | bull | roast | red | brew | protein | 24pk | oz 24pk | starbucks
    dim 4: protein | roast | medium | medium roast | mocha | latte | shake | pods | dark roast | cup
    dim 5: protein | monster | cold brew | brew | ultra | cold | starbucks | monster energy | roast | shake
    dim 6: monster | ultra | alani | monster energy | protein | smart energy | variety pack | alani nu | nu | cold brew
    dim 7: alani | alani nu | nu | slush | smart energy | variety pack | pink slush | cherry slush | smart | protein
    dim 8: ultra | monster | monster energy | ghost | ghost energy | protein | smart energy | cold brew | smart | variety pack
    dim 9: frappuccino | mocha | starbucks | ghost | cold brew | brew | ghost energy | bottles | cold | caramel
    dim 10: ghost | ghost energy | ultra | protein | sour | monster | rockstar | patch kids | sour patch | patch

    sometimes the same word shows up twice like "protein" in dim 4 and 5
    so we printed out the top 10 terms per dim
    in dim 4, we focus on protein-enhanced coffee drinks
    in dim 5, we focus on protein-enhanced energy drinks

    or for "c4" in dim 1 and dim 3
    dim 1 focuses on coffee-style energy drinks
    dim 3 focuses on branded and performance driven energy drinks like red bull
    """
    terms = vectorizer.get_feature_names_out()
    for i, comp in enumerate(svd.components_):
        terms_in_comp = zip(terms, comp)
        sorted_terms = sorted(terms_in_comp, key=lambda x: abs(x[1]), reverse=True)[
            :top_n
        ]
        tags = [term for term, _ in sorted_terms]
        print(f"dim {i}: {' | '.join(tags)}")


# run to compare wiht calebs'
def svd_recommend(
    query, ingredient, products_df, k=100, top_k=5, similarity_penalty=0.5
):
    if ingredient:
        products_df = products_df[
            ~products_df["ingredients"].str.contains(ingredient, case=False)
        ]
    """
    this svd decomposition creates patterns and based on the analysis of top terms in each dimension,
    here explains 30 dims: dimensions 0-29

    each dimension is like a flavor profile, it pulls in words that show up often

    dim 0: energy drink
    dim 1: c4
    dim 2: red bull
    dim 3: c4
    dim 4: protein
    dim 5: protein
    dim 6: monster
    dim 7: alani
    dim 8: ultra
    dim 9: frappuccino
    dim 10: ghost
    dim 11: rockstar
    dim 12: rockstar
    dim 13: mushroom
    dim 14: v8
    dim 15: reign
    dim 16: latte
    dim 17: gfuel
    dim 18: v8
    dim 19: guru
    dim 20: instant
    dim 21: starbucks
    dim 22: starbucks
    dim 23: instant
    dim 24: super coffee
    dim 25: dew
    dim 26: yerba
    dim 27: sang
    dim 28: black rifle
    dim 29: bloom
   
    basically the query is projected on these k=100 dimensions (printed 30 terms)
    + means it matches the dimesions
    - means it opposes it
    we penalize and reward based on the user query
    """
    documents = (
        products_df["description"].fillna("").astype(str)
        + " "
        + products_df["bullet_points"].fillna("").astype(str)
        + " "
        + products_df["flavors"].fillna("").astype(str)
        + " "
        + products_df["reviews"].fillna("").astype(str)
        + " "
        + products_df["whole_title"].fillna("").astype(str)
    )
    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.7, min_df=2, ngram_range=(1, 2), analyzer="word"
    )
    td_matrix = vectorizer.fit_transform(documents)
    svd = TruncatedSVD(n_components=k, random_state=42)
    doc_latent = svd.fit_transform(td_matrix)
    print_svd_tags(vectorizer, svd, top_n=10)
    doc_latent = normalize(doc_latent)
    terms = vectorizer.get_feature_names_out()
    components = svd.components_
    svd_tags = []
    for i, j in enumerate(doc_latent):
        top = j.argsort()[::-1][:10]
        tags = []
        for d in top:
            x = components[d]
            tag = terms[abs(x).argmax()]
            tags.append(tag)
        svd_tags.append(tags)
    products_df = products_df.copy()
    products_df["svd_tags"] = svd_tags
    boosted_query = query.lower()
    if "coffee" in boosted_query:
        boosted_query += " brew beans roast"
    elif "energy" in boosted_query:
        boosted_query += " drink booster"
    elif "juice" in boosted_query:
        boosted_query += " fruit fresh"
    query_vec = vectorizer.transform([boosted_query])
    query_latent = normalize(query_vec.dot(svd.components_.T))
    # calc cossims
    sims = doc_latent.dot(query_latent.T).flatten()
    # based on your query, calc a score
    query_terms = set(query.lower().split())
    for i in range(len(sims)):
        product = products_df.iloc[i]
        product_text = (
            str(product["description"]).lower()
            + " "
            + str(product["flavors"]).lower()
            + " "
            + str(product["whole_title"]).lower()
        )
        # calc the match
        match_score = (
            sum(1 for term in query_terms if term in product_text) / len(query_terms)
            if query_terms
            else 0
        )
        # if it's good match then boost it, otherwise strong penalty
        if match_score > 0.5:
            sims[i] *= 1 + match_score
        else:
            sims[i] *= 0.3
    # remove the same brand or just products that seem very similar
    product_similarities = cosine_similarity(doc_latent)
    final_scores = sims.copy()
    top_N = min(20, len(products_df))
    candidate_indices = sims.argsort()[::-1][:top_N]
    for i in range(1, len(candidate_indices)):
        for j in range(i):
            if product_similarities[candidate_indices[i], candidate_indices[j]] > 0.8:
                final_scores[candidate_indices[i]] *= similarity_penalty
    top_indices = final_scores.argsort()[::-1][:top_k]
    return products_df.iloc[top_indices]
