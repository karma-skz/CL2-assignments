# Word Sense Disambiguation (WSD) - Part 1: Preprocessing

This project implements tokenization, POS tagging, and preprocessing for Word Sense Disambiguation tasks using NLTK.

## Dataset Format

The WSD dataset consists of tab-separated values (TSV) with the following columns:
1. **Sense Key**: WordNet sense key (or `?` for test data)
2. **Lemma**: Word and part of speech (e.g., `keep.v`, `national.a`)
3. **Target Position**: Index of the word to disambiguate
4. **Context**: Tokenized text containing the target word

Example:
```
keep%2:42:07::	keep.v	15	Action by the Committee In pursuance of its mandate...
```

## Features Implemented

### 1. **Data Loading**
- Parses TSV files (training and test data)
- Extracts sense keys, lemmas, target positions, and context sentences
- Handles both labeled training data and blind test data

### 2. **Tokenization**
- **Sentence-level tokenization**: Uses `nltk.sent_tokenize()`
- **Word-level tokenization**: Uses `nltk.word_tokenize()`
- Preserves punctuation and special characters

### 3. **POS Tagging**
- Applies NLTK's averaged perceptron POS tagger
- Labels each token with its part of speech
- Extracts target word's POS tag for disambiguation

### 4. **Preprocessing for WSD**
- **Minimal preprocessing** to preserve context:
  - Lowercasing all tokens
  - Preserving punctuation (important for context)
  - Keeping stopwords (provide contextual information)
  - Maintaining original structure
- Target word preserved for accurate identification

## Installation

### Option 1: Using pip
```bash
pip install -r requirements.txt
```

### Option 2: Direct installation
```bash
pip install nltk
```

The script will automatically download required NLTK resources on first run:
- `punkt` (tokenizer)
- `averaged_perceptron_tagger` (POS tagger)
- `stopwords` (English stopwords)

## Usage

### Basic Usage
Run the main script to process sample data:
```bash
python part1.py
```

This will:
- Process first 5 training examples
- Process first 3 test examples
- Display detailed output for each processed instance
- Show tokenization, POS tags, and preprocessed tokens

### Using the Functions

```python
from part1 import load_wsd_data, process_instance

# Load data
train_data = load_wsd_data('wsd/wsd-data/wsd_train.txt', limit=100)

# Process each instance
processed_data = []
for instance in train_data:
    processed = process_instance(instance)
    processed_data.append(processed)

# Access the results
for item in processed_data[:5]:
    print(f"Target word: {item['target_word']}")
    print(f"POS tag: {item['target_pos_tag']}")
    print(f"Tokens: {item['tokens'][:10]}")
    print()
```

### Processing Custom Text

```python
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Tokenize and tag any text
text = "The bank is near the river bank."
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

print(f"Tokens: {tokens}")
print(f"POS Tags: {pos_tags}")

# Lowercase for preprocessing
preprocessed = [token.lower() for token in tokens]
print(f"Preprocessed: {preprocessed}")
```

## Output Format

Each processed instance contains:
- `sense_key`: WordNet sense key (or `?` for test)
- `lemma`: Word and part of speech
- `target_position`: Index of target word
- `target_word`: The actual target word
- `target_pos_tag`: POS tag of target word
- `context`: Original context sentence
- `tokens`: List of word tokens
- `pos_tags`: List of (word, POS_tag) tuples
- `preprocessed_tokens`: Lowercased tokens

## Example Output

```
======================================================================
Sense Key: keep%2:42:07::
Lemma: keep.v
Target Word: 'keep' at position 15
POS Tag: VB

Context: Action by the Committee In pursuance of its mandate...

Tokens (first 15):
  ['Action', 'by', 'the', 'Committee', 'In', 'pursuance', ...]

POS Tags (first 10):
  [('Action', 'NN'), ('by', 'IN'), ('the', 'DT'), ...]

Preprocessed (first 15):
  ['action', 'by', 'the', 'committee', 'in', 'pursuance', ...]
======================================================================
```

## Dataset Statistics

- **Training data**: 76,050 instances
- **Test data**: 13,302 instances (blind - no sense keys provided)

## POS Tag Set

The implementation uses Penn Treebank POS tags. Common tags include:
- **NN**: Noun, singular
- **NNS**: Noun, plural
- **VB**: Verb, base form
- **VBD**: Verb, past tense
- **JJ**: Adjective
- **RB**: Adverb
- **DT**: Determiner
- **IN**: Preposition

## Design Decisions

### Why Minimal Preprocessing?
For WSD tasks, context is crucial. The preprocessing is intentionally minimal:

1. **Preserving stopwords**: Words like "the", "is", "at" provide important contextual cues
2. **Keeping punctuation**: Helps maintain sentence structure and boundaries
3. **Lowercasing only**: Reduces vocabulary size without losing meaning
4. **No stemming/lemmatization of context**: Preserves word forms that may be discriminative

### Target Word Handling
The target word position is carefully tracked to ensure:
- Accurate identification in the tokenized sequence
- Extraction of its POS tag for feature engineering
- Preservation in preprocessing pipeline

## Next Steps

After preprocessing, typical WSD pipelines involve:
1. Feature extraction (context windows, collocations, etc.)
2. Building a classification model (Naive Bayes, Neural Networks)
3. Training on labeled data
4. Predicting sense keys for test data

## References

- NLTK Documentation: https://www.nltk.org/
- WordNet: https://wordnet.princeton.edu/

---
The overlap words come from both the definition AND the examples. The output shows only the definition but the algorithm correctly uses the full gloss (definition + examples).

In the output:

Definition shown: "keep in a certain state, position, or activity; e.g.,"
But examples are also used (not shown): "keep clean", "hold in place", "She always held herself as a lady", "The students keep me on my toes"
That's where "the" and extra "keep" come from. The overlap calculation is correct - the output just doesn't display the examples to keep it concise. The comment in the code says "Get gloss words (definition + examples)" which is accurate.

---