import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, Counter
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)


def load_wsd_data(filepath, limit=None):
    # Load WSD data from TSV file
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) == 4:
                sense_key, lemma, position, context = parts
                data.append({
                    'sense_key': sense_key,
                    'lemma': lemma,
                    'target_position': int(position),
                    'context': context
                })
            if limit and len(data) >= limit:
                break
    return data


def process(instance):
    context = instance['context']
    target_pos = instance['target_position']
    
    tokens = word_tokenize(context)
    pos_tags = pos_tag(tokens)
    preprocessed = [token.lower() for token in tokens] # otherwise word positions muight change
    
    target_word = tokens[target_pos] if target_pos < len(tokens) else None
    target_pos_tag = pos_tags[target_pos][1] if target_pos < len(pos_tags) else None
    
    return {
        'sense_key': instance['sense_key'],
        'lemma': instance['lemma'],
        'target_position': target_pos,
        'target_word': target_word,
        'target_pos_tag': target_pos_tag,
        'context': context,
        'tokens': tokens,
        'pos_tags': pos_tags,
        'preprocessed_tokens': preprocessed
    }

# TASK 2: LIST POSSIBLE SENSES FOR CONTENT WORDS
def penn_to_wn_pos(penn_tag):
    """Convert Penn Treebank POS tag to WordNet POS tag."""
    if penn_tag.startswith('N'):
        return wn.NOUN
    elif penn_tag.startswith('V'):
        return wn.VERB
    elif penn_tag.startswith('R'):
        return wn.ADV
    elif penn_tag.startswith('J'):
        return wn.ADJ
    return None


def list_word_senses(word, pos_tag):
    """Get all WordNet senses for a word with given POS tag."""
    wn_pos = penn_to_wn_pos(pos_tag)
    if not wn_pos:
        return []
    
    synsets = wn.synsets(word, pos=wn_pos)
    senses = []
    for synset in synsets:
        senses.append({
            'synset': synset.name(), #type: ignore
            'definition': synset.definition(), #type: ignore
            'examples': synset.examples() #type: ignore
        })
    return senses


def extract_content_words_from_sentences(processed_data, num_sentences=10):
    """Extract content words (N, V, J, R) from first N sentences."""
    content_words = {}
    sentence_count = 0
    
    for instance in processed_data:
        if sentence_count >= num_sentences:
            break
        sentence_count += 1
        
        # Get content words from this sentence
        for token, pos_tag in instance['pos_tags']:
            if penn_to_wn_pos(pos_tag):  # Check if it's a content word
                word_lower = token.lower()
                key = f"{word_lower}_{pos_tag[:2]}"  # e.g., "keep_VB"
                
                if key not in content_words:
                    senses = list_word_senses(word_lower, pos_tag)
                    if senses:  # Only add if WordNet has senses
                        content_words[key] = {
                            'word': word_lower,
                            'pos': pos_tag,
                            'senses': senses
                        }
    
    return content_words

# TASK 3: MOST FREQUENT SENSE BASELINE
def get_mfs(word, pos_tag):
    """Get most frequent sense (first synset) from WordNet."""
    wn_pos = penn_to_wn_pos(pos_tag)
    if not wn_pos:
        return None
    
    synsets = wn.synsets(word, pos=wn_pos)
    return synsets[0].name() if synsets else None #type: ignore


def evaluate_mfs_baseline(processed_data):
    """Evaluate MFS baseline on training data."""
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    for instance in processed_data:
        # Skip test data (sense_key = '?')
        if instance['sense_key'] == '?':
            continue
        
        word = instance['target_word'].lower()
        pos = instance['target_pos_tag']
        
        # Get MFS prediction
        pred = get_mfs(word, pos)
        
        # Get gold synset using lemma_from_key (correct method)
        gold = None
        try:
            if instance['sense_key']:
                lemma = wn.lemma_from_key(instance['sense_key'])
                gold = lemma.synset().name() if lemma else None
        except Exception:
            gold = None
        
        if pred:
            total += 1
            # Compare full synset names (e.g., "keep.v.01" == "keep.v.06")
            is_correct = (pred == gold)
            if is_correct:
                correct += 1
            
            y_true.append(1)  # Always 1 (true label exists)
            y_pred.append(1 if is_correct else 0)  # 1 if correct, 0 if wrong
    
    precision = recall = f1 = correct / total if total > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'correct': correct, 'total': total, 
            'y_true': y_true, 'y_pred': y_pred}


# TASK 4: SIMPLIFIED LESK ALGORITHM
def get_context_words(instance, window_size=None):
    """Get context words around target word (or full sentence if window_size=None)."""
    tokens = instance['preprocessed_tokens']
    if window_size is None:
        return set(tokens)
    
    target_pos = instance['target_position']
    start = max(0, target_pos - window_size)
    end = min(len(tokens), target_pos + window_size + 1)
    return set(tokens[start:end])


def simplified_lesk(word, pos_tag, context_words):
    """
    Simplified Lesk: Find sense with maximum overlap between gloss and context.
    No lemmatization or stopword removal (basic version).
    Returns (best_synset, max_overlap, overlap_details).
    """
    wn_pos = penn_to_wn_pos(pos_tag)
    if not wn_pos:
        return None, 0, {}
    
    # POS-based filtering: only get senses matching the POS
    synsets = wn.synsets(word, pos=wn_pos)
    if not synsets:
        return None, 0, {}
    
    max_overlap = 0
    best_synset = None
    overlap_details = {}
    
    for synset in synsets:
        # Get gloss words (definition + examples)
        gloss = synset.definition() #type: ignore
        for example in synset.examples(): #type: ignore
            gloss += " " + example
        
        gloss_words = set(word_tokenize(gloss.lower()))
        
        # Calculate overlap
        overlap = len(context_words & gloss_words)
        overlap_details[synset.name()] = { #type: ignore
            'overlap': overlap,
            'common_words': list(context_words & gloss_words)
        }
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_synset = synset
    
    return best_synset, max_overlap, overlap_details


def improved_lesk(word, pos_tag, context_words):
    """
    Improved Lesk: Adds lemmatization and stopword removal to basic Lesk.
    Returns (best_synset, max_overlap, overlap_details).
    """
    wn_pos = penn_to_wn_pos(pos_tag)
    if not wn_pos:
        return None, 0, {}
    
    # POS-based filtering: only get senses matching the POS
    synsets = wn.synsets(word, pos=wn_pos)
    if not synsets:
        return None, 0, {}
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Lemmatize and remove stopwords from context
    context_filtered = set()
    for w in context_words:
        if w not in stop_words:
            lemma = lemmatizer.lemmatize(w, pos=wn_pos)
            context_filtered.add(lemma)
    
    max_overlap = 0
    best_synset = None
    overlap_details = {}
    
    for synset in synsets:
        # Get gloss words (definition + examples)
        gloss = synset.definition() #type: ignore
        for example in synset.examples(): #type: ignore
            gloss += " " + example
        
        gloss_tokens = word_tokenize(gloss.lower())
        
        # Lemmatize and remove stopwords from gloss
        gloss_filtered = set()
        for w in gloss_tokens:
            if w not in stop_words:
                lemma = lemmatizer.lemmatize(w, pos=wn_pos)
                gloss_filtered.add(lemma)
        
        # Calculate overlap
        overlap = len(context_filtered & gloss_filtered)
        overlap_details[synset.name()] = { #type: ignore
            'overlap': overlap,
            'common_words': list(context_filtered & gloss_filtered)
        }
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_synset = synset
    
    return best_synset, max_overlap, overlap_details


def evaluate_lesk(processed_data, window_size=None, use_improved=False):
    """Evaluate Lesk on training data. Set use_improved=True for improved version."""
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    lesk_func = improved_lesk if use_improved else simplified_lesk
    
    for instance in processed_data:
        if instance['sense_key'] == '?':
            continue
        
        word = instance['target_word'].lower()
        pos = instance['target_pos_tag']
        
        context_words = get_context_words(instance, window_size)
        best_synset, overlap, _ = lesk_func(word, pos, context_words)
        
        # Get prediction
        pred = best_synset.name() if best_synset else None
        
        # Get gold synset using lemma_from_key (correct method)
        gold = None
        try:
            if instance['sense_key']:
                lemma = wn.lemma_from_key(instance['sense_key'])
                gold = lemma.synset().name() if lemma else None
        except Exception:
            gold = None
        
        if pred:
            total += 1
            # Compare full synset names (e.g., "keep.v.01" == "keep.v.06")
            is_correct = (pred == gold)
            if is_correct:
                correct += 1
            
            y_true.append(1)
            y_pred.append(1 if is_correct else 0)
    
    precision = recall = f1 = correct / total if total > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'correct': correct, 'total': total,
            'y_true': y_true, 'y_pred': y_pred}


def show_lesk_example(processed_instance):
    # Show detailed Lesk example for one word.
    word = processed_instance['target_word'].lower()
    pos = processed_instance['target_pos_tag']
    context_words = get_context_words(processed_instance, window_size=5)
    
    print(f"\n{'='*70}")
    print(f"LESK ALGORITHM EXAMPLE")
    print(f"{'='*70}")
    print(f"Target Word: '{word}' (POS: {pos})")
    print(f"Gold Sense: {processed_instance['sense_key']}")
    print(f"Context (±5 words): {list(context_words)[:15]}...")
    
    best_synset, max_overlap, overlap_details = simplified_lesk(word, pos, context_words)
    
    for synset_name, details in list(overlap_details.items())[:5]:  # Show top 5
        synset = wn.synset(synset_name)
        print(f"\n  {synset_name}:")
        print(f"    Definition: {synset.definition()}") #type: ignore
        print(f"    Overlap: {details['overlap']} words")
        print(f"    Common: {details['common_words'][:10]}")
    
    print(f"\n  SELECTED: {best_synset.name() if best_synset else 'None'}")
    print(f"  Max Overlap: {max_overlap}")
    print(f"{'='*70}\n")


# NAIVE BAYES
def extract_features(instance, window_size=5):
    """
    Extract features for Naive Bayes classifier:
    - Context words in a window around target
    - Collocations (word before, word after)
    - POS tags in context
    """
    target_pos = instance['target_position']
    tokens = instance['preprocessed_tokens']
    pos_tags = instance['pos_tags']
    
    features = {}
    
    # Context words in window
    start = max(0, target_pos - window_size)
    end = min(len(tokens), target_pos + window_size + 1)
    
    for i in range(start, end):
        if i != target_pos:  # Skip target word itself
            features[f'context_{tokens[i]}'] = 1
    
    # Collocations
    if target_pos > 0:
        features[f'word_before_{tokens[target_pos - 1]}'] = 1
        if target_pos > 1:
            features[f'word_before2_{tokens[target_pos - 2]}'] = 1
    
    if target_pos < len(tokens) - 1:
        features[f'word_after_{tokens[target_pos + 1]}'] = 1
        if target_pos < len(tokens) - 2:
            features[f'word_after2_{tokens[target_pos + 2]}'] = 1
    
    # POS tags in context
    for i in range(start, end):
        if i != target_pos and i < len(pos_tags):
            features[f'pos_{pos_tags[i][1]}'] = 1
    
    # Bigrams in context
    for i in range(start, end - 1):
        if i != target_pos and i + 1 != target_pos:
            features[f'bigram_{tokens[i]}_{tokens[i+1]}'] = 1
    
    return features


def train_naive_bayes(train_data):
    """
    Train a Naive Bayes classifier for each ambiguous word.
    Returns a dictionary of {word: (classifier, vectorizer, label_to_synset)}
    """
    # Group instances by target word
    word_instances = defaultdict(list)
    
    for instance in train_data:
        if instance['sense_key'] == '?':
            continue
        
        word = instance['target_word'].lower()
        
        # Get gold synset
        try:
            lemma = wn.lemma_from_key(instance['sense_key'])
            gold = lemma.synset().name() if lemma else None
        except Exception:
            gold = None
        
        if gold:
            word_instances[word].append((instance, gold))
    
    # Train a classifier for each word that has multiple senses
    classifiers = {}
    
    for word, instances in word_instances.items():
        # Only train if word has at least 2 different senses and at least 10 examples
        sense_counts = Counter([synset for _, synset in instances])
        if len(sense_counts) < 2 or len(instances) < 10:
            continue
        
        # Extract features and labels
        X_features = []
        y_labels = []
        
        for inst, synset in instances:
            features = extract_features(inst)
            X_features.append(features)
            y_labels.append(synset)
        
        # Vectorize features
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(X_features)
        
        # Train Naive Bayes
        clf = MultinomialNB(alpha=1.0)
        clf.fit(X, y_labels)
        
        classifiers[word] = {
            'classifier': clf,
            'vectorizer': vectorizer,
            'senses': list(sense_counts.keys())
        }
    
    return classifiers


def predict_naive_bayes(instance, classifiers):
    """Predict sense using Naive Bayes classifier."""
    word = instance['target_word'].lower()
    
    # Check if we have a trained classifier for this word
    if word not in classifiers:
        return None
    
    model = classifiers[word]
    
    # Extract features
    features = extract_features(instance)
    X = model['vectorizer'].transform([features])
    
    # Predict
    pred = model['classifier'].predict(X)[0]
    
    return pred


def evaluate_naive_bayes(train_data, test_data):
    """Train on train_data and evaluate on test_data (both should have gold labels)."""
    print("\nTraining Naive Bayes classifiers...")
    classifiers = train_naive_bayes(train_data)
    print(f"Trained classifiers for {len(classifiers)} ambiguous words")
    
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    for instance in test_data:
        if instance['sense_key'] == '?':
            continue
        
        # Get gold synset
        try:
            lemma = wn.lemma_from_key(instance['sense_key'])
            gold = lemma.synset().name() if lemma else None
        except Exception:
            gold = None
        
        if not gold:
            continue
        
        # Predict
        pred = predict_naive_bayes(instance, classifiers)
        
        if pred:
            total += 1
            is_correct = (pred == gold)
            if is_correct:
                correct += 1
            
            y_true.append(1)
            y_pred.append(1 if is_correct else 0)
    
    precision = recall = f1 = correct / total if total > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct,
        'total': total,
        'y_true': y_true,
        'y_pred': y_pred,
        'num_classifiers': len(classifiers)
    }


# ============================================================================
# MAIN EXECUTION FOR ALL TASKS
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    os.makedirs('part1_plots', exist_ok=True)
    
    # Load and process training data
    train_raw = load_wsd_data('wsd/wsd-data/wsd_train.txt')
    train_processed = [process(inst) for inst in train_raw]
    
    # Split training data 70-30 for Naive Bayes
    split_idx = int(len(train_processed) * 0.7)
    train_70 = train_processed[:split_idx]
    test_30 = train_processed[split_idx:]
    
    print(f"Loaded {len(train_processed)} training instances")
    print(f"Split: {len(train_70)} training (70%), {len(test_30)} test (30%)")
    
    # TASK 2: Extract content words from first 10 sentences
    content_words = extract_content_words_from_sentences(train_processed, num_sentences=10)
    
    # Save to JSON file
    with open('word_senses.json', 'w') as f:
        json.dump(content_words, f, indent=2)
    print(f"\nTask 2: Saved {len(content_words)} content words with senses to 'word_senses.json'")
    
    # TASK 3: Evaluate MFS baseline on 30% test set
    mfs_results = evaluate_mfs_baseline(test_30)
    print(f"\nTask 3 - MFS Baseline: Precision={mfs_results['precision']:.4f}, Recall={mfs_results['recall']:.4f}, F1={mfs_results['f1']:.4f}")
    
    # Create confusion matrix for MFS
    cm_mfs = confusion_matrix(mfs_results['y_true'], mfs_results['y_pred'], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_mfs, display_labels=['Incorrect', 'Correct'])
    disp.plot(cmap='Blues')
    plt.title('MFS Baseline - Confusion Matrix')
    plt.savefig('part1_plots/mfs_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved MFS confusion matrix to 'part1_plots/mfs_confusion_matrix.png'")
    
    # TASK 4: Show Lesk example and evaluate on 30% test set
    show_lesk_example(train_processed[0])
    lesk_results = evaluate_lesk(test_30, window_size=None, use_improved=False)
    print(f"\nTask 4 - Simplified Lesk: Precision={lesk_results['precision']:.4f}, Recall={lesk_results['recall']:.4f}, F1={lesk_results['f1']:.4f}")
    
    # Create confusion matrix for Lesk
    cm_lesk = confusion_matrix(lesk_results['y_true'], lesk_results['y_pred'], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_lesk, display_labels=['Incorrect', 'Correct'])
    disp.plot(cmap='Greens')
    plt.title('Simplified Lesk - Confusion Matrix')
    plt.savefig('part1_plots/lesk_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Lesk confusion matrix to 'part1_plots/lesk_confusion_matrix.png'")
    
    # Evaluate Improved Lesk on 30% test set
    improved_results = evaluate_lesk(test_30, window_size=None, use_improved=True)
    print(f"\nImproved Lesk: Precision={improved_results['precision']:.4f}, Recall={improved_results['recall']:.4f}, F1={improved_results['f1']:.4f}")
    
    # Create confusion matrix for Improved Lesk
    cm_improved = confusion_matrix(improved_results['y_true'], improved_results['y_pred'], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_improved, display_labels=['Incorrect', 'Correct'])
    disp.plot(cmap='Purples')
    plt.title('Improved Lesk - Confusion Matrix')
    plt.savefig('part1_plots/improved_lesk_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Improved Lesk confusion matrix to 'part1_plots/improved_lesk_confusion_matrix.png'")
    
    # TASK 5: Naive Bayes WSD (train on 70%, test on 30%)
    nb_results = evaluate_naive_bayes(train_70, test_30)
    print(f"\nNaive Bayes: Precision={nb_results['precision']:.4f}, Recall={nb_results['recall']:.4f}, F1={nb_results['f1']:.4f}")
    print(f"  Trained {nb_results['num_classifiers']} word-specific classifiers")
    
    # Create confusion matrix for Naive Bayes
    if len(nb_results['y_true']) > 0:
        cm_nb = confusion_matrix(nb_results['y_true'], nb_results['y_pred'], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['Incorrect', 'Correct'])
        disp.plot(cmap='Oranges')
        plt.title('Naive Bayes - Confusion Matrix')
        plt.savefig('part1_plots/naive_bayes_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved Naive Bayes confusion matrix to 'part1_plots/naive_bayes_confusion_matrix.png'")
    
    # Comparison
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON (30% Test Set Performance):")
    print(f"{'='*70}")
    print(f"MFS Baseline:      F1={mfs_results['f1']:.4f} (Precision={mfs_results['precision']:.4f}, Recall={mfs_results['recall']:.4f})")
    print(f"Simplified Lesk:   F1={lesk_results['f1']:.4f} (Precision={lesk_results['precision']:.4f}, Recall={lesk_results['recall']:.4f})")
    print(f"Improved Lesk:     F1={improved_results['f1']:.4f} (Precision={improved_results['precision']:.4f}, Recall={improved_results['recall']:.4f})")
    print(f"Naive Bayes:       F1={nb_results['f1']:.4f} (Precision={nb_results['precision']:.4f}, Recall={nb_results['recall']:.4f})")
    print(f"{'='*70}\n")
