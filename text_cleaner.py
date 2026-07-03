import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re

def clean_string(s):
    if not isinstance(s, str):
        return ""
    # Remove excessive whitespace, normalize formatting
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def resolve_text_inconsistencies(series: pd.Series, similarity_threshold: float = 0.8) -> tuple[pd.Series, dict]:
    """
    Finds lexically similar categorical values using TF-IDF + Cosine Similarity and standardizes them.
    Returns:
        - The cleaned Series.
        - A dictionary of mappings: {original_value: corrected_value}.
    """
    # 1. Clean basic whitespace
    cleaned_series = series.astype(str).apply(clean_string)
    
    # Value counts to determine the most frequent spelling (the centroid)
    val_counts = cleaned_series.value_counts()
    unique_vals = list(val_counts.index)
    
    if len(unique_vals) <= 1:
        return cleaned_series, {}

    # 2. Build char n-gram TF-IDF vectorizer (handles typos well)
    # We use 2-3 char n-grams to capture substrings and spelling mistakes
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(unique_vals)
    except ValueError:
        # If vocabulary is empty (e.g., all strings are too short or empty)
        return cleaned_series, {}
        
    # 3. Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 4. Group similar items (Connected Components)
    visited = set()
    mappings = {}
    
    for i in range(len(unique_vals)):
        if i in visited:
            continue
            
        # Find all items similar to item i
        similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]
        
        if len(similar_indices) > 1:
            # We have a cluster of similar strings!
            cluster_vals = [unique_vals[idx] for idx in similar_indices]
            
            # Find the most frequent spelling (centroid) in the cluster
            centroid = max(cluster_vals, key=lambda x: val_counts.get(x, 0))
            
            # Map everything in the cluster to the centroid
            for val in cluster_vals:
                if val != centroid:
                    mappings[val] = centroid
                    
            # Mark all these indices as visited
            for idx in similar_indices:
                visited.add(idx)
        else:
            visited.add(i)
            
    # Apply mapping
    if mappings:
        cleaned_series = cleaned_series.replace(mappings)
        
    return cleaned_series, mappings

if __name__ == "__main__":
    # Test text cleaner
    dirty_cats = pd.Series([
        "Google Inc.", "google inc", "Google", "Microsoft Corp.", 
        "microsoft corp", "Microsoft", "Apple Inc", "Apple", "Apple Inc.",
        "Amazon.com", "Amazon"
    ])
    cleaned, maps = resolve_text_inconsistencies(dirty_cats, similarity_threshold=0.75)
    print("Cleaned Series:")
    print(cleaned.values)
    print("\nMappings:")
    for k, v in maps.items():
        print(f"  '{k}' -> '{v}'")
