import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import gc  # For garbage collection

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)

# --- Configuration ---
CSV_FILE = 'train.csv'
# Constraint: Ignore the class indices in the data; only process the sentences.
NUM_SENTENCES = 15000  # Reduced to 15k to manage memory (Original: 120k)
SVD_K = 100            # Number of components for SVD
WINDOW_SIZES = [1, 2, 3, 4]

# --- Section 2.1: Data Preprocessing ---
def load_and_preprocess_data(filepath, limit):
    """
    Loads the dataset and limits the number of rows to manage memory usage.
    Constraint: Ignore the class indices in the data.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # Constraint: Ignore class indices, only process sentences (Description column)
        # Slice the dataframe to reduce memory footprint
        df = df[0:limit] 
        print(f"Data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the file is in the directory.")
        return None

# --- Section 2.1: Co-occurrence Matrix Construction ---
def build_cooccurrence_matrix(df, window_size):
    """
    Constructs a co-occurrence matrix for the given window size.
    Rows: Target words, Columns: Context words.
    """
    print(f"Building co-occurrence matrix for window size {window_size}...")
    
    # Extract vocabulary
    all_tokens = []
    for text in df['Description']:
        tokens = word_tokenize(text.lower())
        all_tokens.extend(tokens)
    
    unique_tokens = list(set(all_tokens))
    token_to_index = {token: i for i, token in enumerate(unique_tokens)}
    vocab_size = len(unique_tokens)
    
    print(f"Vocabulary Size: {vocab_size}")
    
    # Initialize Matrix
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=int)
    
    # Populate Matrix based on frequency of co-occurrence
    for text in df['Description']:
        tokens = word_tokenize(text.lower())
        for i, target in enumerate(tokens):
            target_index = token_to_index[target]
            
            # Define window boundaries
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            
            for j in range(start, end):
                if i != j: # Skip self
                    context = tokens[j]
                    context_index = token_to_index[context]
                    cooccurrence_matrix[target_index][context_index] += 1
                    
    return cooccurrence_matrix, unique_tokens

# --- Section 2.2: Dimensionality Reduction ---
def perform_svd(cooccurrence_df, k):
    """
    Applies Singular Value Decomposition (SVD) to reduce dimensionality.
    Returns the U matrix as a DataFrame (dense, low-dimensional word embeddings).
    """
    print(f"Performing SVD with k={k}...")
    # Convert to float for SVD calculation
    matrix_np = cooccurrence_df.to_numpy(dtype=np.float64)
    
    # U: Unitary matrix (Word Embeddings), Sigma: Singular values, VT: Transpose of V
    U, Sigma, VT = svds(matrix_np, k=k)
    
    # Create DataFrame for easy lookup
    U_df = pd.DataFrame(U, index=cooccurrence_df.index)
    return U_df

# --- Section 2.3: Similarity Evaluation ---
def calculate_cosine_similarity(vec1, vec2):
    """Calculates Cosine Similarity (1 - Cosine Distance)."""
    return 1 - cosine(vec1, vec2)

def evaluate_word_pairs(U_df, window_size):
    """
    Calculates and prints similarity scores for specific test pairs.
    """
    print(f"Testing for Window Size {window_size}")
    
    # Word Pair Selection (Section 2.3)
    word_pairs = [
        ('space', 'planet'),            # Related concepts
        ('astronauts', 'robot'),        # Related but distinct entities
        ('more', 'less'),               # Antonyms
        ('scientists', 'researchers'),  # Synonyms
        ('rocket', 'telescope')         # Different roles in same domain
    ]
    
    results = []
    for word1, word2 in word_pairs:
        try:
            vec1 = U_df.loc[word1]
            vec2 = U_df.loc[word2]
            score = calculate_cosine_similarity(vec1, vec2)
            results.append((word1, word2, score))
            print(f"Cosine Similarity between '{word1}' and '{word2}': {score:.4f}")
        except KeyError as e:
            print(f"Word {e} not found in the vocabulary.")
    return results

# --- Section 2.4: Experimentation ---
def main():
    # Load Data (Section 2.1)
    df = load_and_preprocess_data(CSV_FILE, NUM_SENTENCES)
    if df is None:
        return

    # Iterate through different window sizes (Section 2.4)
    for window_size in WINDOW_SIZES:
        print("-" * 50)
        
        # 1. Build Matrix (Section 2.1)
        matrix, tokens = build_cooccurrence_matrix(df, window_size)
        
        # Convert to DataFrame for easier handling in SVD function
        cooccurrence_df = pd.DataFrame(matrix, index=tokens, columns=tokens)
        
        # 2. Perform SVD (Section 2.2)
        U_df = perform_svd(cooccurrence_df, SVD_K)
        
        # 3. Evaluate (Section 2.3)
        evaluate_word_pairs(U_df, window_size)
        
        # 4. Memory Management
        # Explicitly deleting large objects to prevent RAM overflow
        del matrix
        del cooccurrence_df
        del U_df
        gc.collect() 
        print(f"Buffer cleared after window size {window_size}")

if __name__ == "__main__":
    main()