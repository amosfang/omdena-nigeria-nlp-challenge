from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib

import spacy
import re

custom_stop_words = []

# Load SciSpaCy
nlp = spacy.load("en_core_sci_lg", disable=["parser"])

batch_size = 500

def preprocess_text(df, text_col="symptoms", custom_stop_words=custom_stop_words, batch_size=500):
    """
    Function to clean text by converting digits to words, removing punctuation/stopwords, and lemmatizing.
    """

    # Convert to lowercase
    df[text_col] = df[text_col].str.lower()

    # Add custom stop words to SpaCy's stop words list
    for word in custom_stop_words:
        nlp.vocab[word].is_stop = True

    # Step 1: Process text using SpaCy pipeline
    nlp_pipe = nlp.pipe(df[text_col], batch_size=batch_size, disable=["parser"])

    tokens = []
    entities = []

    for doc in nlp_pipe:
        # Remove punctuation & stop words, then lemmatize
        filtered_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]

        named_entities = [ent.text.strip() for ent in doc.ents]

        tokens.append(filtered_tokens)
        entities.append(named_entities)

    # Remove custom stop words from tokens
    tokens = [[token for token in doc if token not in custom_stop_words] for doc in tokens]

    # Join tokens into a cleaned text column
    df["processed_symptoms"] = [' '.join(token) for token in tokens]
    df["named_entities"] = entities

    return df
  
# Custom Preprocessor Class
class Preprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting required for this preprocessor
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            # Convert Series to DataFrame
            df = X.to_frame(name="Review")
        elif isinstance(X, np.ndarray):
            # Convert NumPy array to DataFrame
            df = pd.DataFrame(X, columns=["Review"])
        else:
            raise ValueError("Input X must be a pandas Series or NumPy array.")

        # Call the preprocess_text function for text cleaning
        preprocess_text(df, text_col="symptoms", custom_stop_words=custom_stop_words)
        
        # Return a pandas Series containing list of named entities for each disease
        return df["named_entitites"]
