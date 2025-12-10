import csv, math, nltk
from collections import defaultdict, Counter
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------
# Load GAP dataset
# -------------------------
def load_gap_dataset(path):
    data = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

# -------------------------
# Candidate extraction
# -------------------------
def extract_candidates(text, pron_offset):
    sentences = nltk.sent_tokenize(text)
    candidates = []
    char_count = 0
    pron_sentence_index = None

    # Find sentence of pronoun
    for i, sent in enumerate(sentences):
        if char_count <= pron_offset < char_count + len(sent):
            pron_sentence_index = i
            break
        char_count += len(sent) + 1

    if pron_sentence_index is None:
        pron_sentence_index = 0

    # Extract NPs from all sentences up to pronoun sentence
    for i, sent in enumerate(sentences[:pron_sentence_index + 1]):
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
        parser = RegexpParser(grammar)
        tree = parser.parse(tagged)

        token_index = 0
        for subtree in tree:
            if hasattr(subtree, 'label') and subtree.label() == 'NP': # type: ignore
                np_text = ' '.join([w for w, t in subtree.leaves()]) # type: ignore
                # Approximate character offset
                offset = char_count + sum(len(t)+1 for t in tokens[:token_index])
                if i < pron_sentence_index or offset < pron_offset:
                    candidates.append((np_text, offset, i))
            token_index += len(subtree.leaves()) if hasattr(subtree, 'leaves') else 1 # type: ignore

        char_count += len(sent) + 1

    return candidates, pron_sentence_index

# -------------------------
# Gender / number helpers
# -------------------------
def get_gender_from_pronoun(pron):
    w = pron.lower()
    if w in {'he','him','his'}: return 'male'
    if w in {'she','her','hers'}: return 'female'
    if w in {'they','them','their'}: return 'plural'
    return 'neutral'

def is_plural_noun(word):
    if not word: return False
    tag = pos_tag([word])[0][1]
    return tag in ['NNS','NNPS']

# -------------------------
# Feature extraction
# -------------------------
def extract_features(pron, pron_offset, pron_sentence_index, candidate, cand_offset, cand_sentence_index):
    feats = {}
    
    # Distance features
    token_distance = abs(pron_offset - cand_offset)
    feats['token_distance'] = token_distance
    if token_distance < 35:
        feats['distance_bin'] = 'close'
    elif token_distance < 150:
        feats['distance_bin'] = 'medium'
    else:
        feats['distance_bin'] = 'far'

    feats['same_sentence'] = int(pron_sentence_index == cand_sentence_index)

    # Gender and number agreement
    pron_gender = get_gender_from_pronoun(pron)
    feats['pron_gender'] = pron_gender
    cand_words = candidate.split()
    cand_is_plural = is_plural_noun(cand_words[-1]) if cand_words else False
    pron_is_plural = pron_gender == 'plural'
    feats['number_agree'] = int(pron_is_plural == cand_is_plural)
    feats['gender_agree'] = int((pron_gender in {'male','female'}) and cand_words[0][0].isupper())

    # Rule-based tie-breakers
    feats['boost_same_sentence'] = int(pron_sentence_index == cand_sentence_index)
    feats['boost_close_distance'] = int(token_distance < 20)

    # Head match
    feats['head_match'] = int(cand_words[-1].lower() in pron.lower())

    return feats

# -------------------------
# Prepare data
# -------------------------
def prepare_data(rows):
    X, y, meta = [], [], []
    for r in rows:
        text = r['Text']
        pron = r['Pronoun']
        po = int(r['Pronoun-offset'])
        
        # Extract candidates
        candidates, pron_sent_idx = extract_candidates(text, po)

        # True answer
        true_answer = None
        if r.get('A-coref','').strip().upper() == 'TRUE':
            true_answer = r['A']
        elif r.get('B-coref','').strip().upper() == 'TRUE':
            true_answer = r['B']

        # Create training examples
        for cand, cand_offset, cand_sent_idx in candidates:
            feats = extract_features(pron, po, pron_sent_idx, cand, cand_offset, cand_sent_idx)
            X.append(feats)
            is_correct = 1 if true_answer and cand.lower() == true_answer.lower() else 0
            y.append(is_correct)
            meta.append({
                'id': r.get('ID',''),
                'text': text,
                'pronoun': pron,
                'candidate': cand,
                'true_answer': true_answer,
                'is_correct': bool(is_correct)
            })
    return X, y, meta

# -------------------------
# Train Naive Bayes
# -------------------------
def train_nb(X, y):
    feature_values = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    class_counts = defaultdict(int)
    feature_names = list(X[0].keys()) if X else []
    all_feature_values = defaultdict(set)

    for feats in X:
        for f in feature_names:
            all_feature_values[f].add(feats[f])

    for feats, label in zip(X, y):
        class_counts[label] += 1
        for f in feature_names:
            v = feats[f]
            feature_values[f][v][label] += 1

    return {
        'feature_values': feature_values,
        'class_counts': class_counts,
        'feature_names': feature_names,
        'all_feature_values': all_feature_values
    }

# -------------------------
# Predict using NB
# -------------------------
def predict_nb(feats, model):
    logprob = {}
    total = sum(model['class_counts'].values())

    for c in model['class_counts']:
        logprob[c] = math.log(model['class_counts'][c] / total)
        for f in model['feature_names']:
            v = feats[f]
            num_values = len(model['all_feature_values'][f])
            count_fv_c = model['feature_values'][f][v][c]
            total_c = model['class_counts'][c]
            p = (count_fv_c + 1) / (total_c + num_values)
            logprob[c] += math.log(p)

    return logprob[1] - logprob[0]

# -------------------------
# Evaluate NB
# -------------------------
def evaluate_nb(model, test_rows, save_examples=False):
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    # Store examples for each category
    tp_examples = []
    fp_examples = []
    fn_examples = []
    tn_examples = []

    for r in test_rows:
        text = r['Text']
        pron = r['Pronoun']
        po = int(r['Pronoun-offset'])

        # True answer
        true_answer = None
        if r.get('A-coref','').strip().upper() == 'TRUE':
            true_answer = r['A']
        elif r.get('B-coref','').strip().upper() == 'TRUE':
            true_answer = r['B']

        if not true_answer:
            continue

        candidates, pron_sent_idx = extract_candidates(text, po)
        if not candidates:
            false_negatives += 1
            total += 1
            fn_examples.append({
                'id': r.get('ID', ''),
                'text': text,
                'pronoun': pron,
                'true_answer': true_answer,
                'predicted': None,
                'reason': 'No candidates extracted'
            })
            continue

        # Score candidates
        best_score = float('-inf')
        best_candidate = None
        all_scores = []
        for cand, cand_offset, cand_sent_idx in candidates:
            feats = extract_features(pron, po, pron_sent_idx, cand, cand_offset, cand_sent_idx)
            score = predict_nb(feats, model)
            all_scores.append((cand, score, feats))
            if score > best_score:
                best_score = score
                best_candidate = cand

        # Evaluate
        example_info = {
            'id': r.get('ID', ''),
            'text': text,
            'pronoun': pron,
            'true_answer': true_answer,
            'predicted': best_candidate,
            'score': best_score,
            'all_candidates': all_scores
        }
        
        if best_candidate and best_candidate.lower() == true_answer.lower():
            correct += 1
            true_positives += 1
            tp_examples.append(example_info)
        else:
            if best_candidate:
                false_positives += 1
                fp_examples.append(example_info)
            false_negatives += 1
            fn_examples.append(example_info)
        total += 1

    acc = correct / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives+false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives+false_negatives) > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0

    print("\n=== Evaluation ===")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"\n--- Confusion Matrix ---")
    print(f"                  Predicted")
    print(f"                Positive  Negative")
    print(f"Actual Positive    {true_positives:4d}     {false_negatives:4d}")
    print(f"Actual Negative    {false_positives:4d}     {true_negatives:4d}")
    print(f"\nTrue Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"True Negatives:  {true_negatives}")
    print(f"\n--- Metrics ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    if save_examples:
        save_classification_examples(tp_examples, fp_examples, fn_examples, tn_examples)
    
    return acc, precision, recall, f1

def save_classification_examples(tp_examples, fp_examples, fn_examples, tn_examples):
    """Save top 25 examples from each classification category"""
    with open('nb_examples_detailed.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("NAIVE BAYES CLASSIFICATION EXAMPLES\n")
        f.write("="*80 + "\n\n")
        
        # True Positives
        f.write("="*80 + "\n")
        f.write("TRUE POSITIVES (TP): Correctly Predicted Antecedent\n")
        f.write(f"Total: {len(tp_examples)}, Showing: {min(25, len(tp_examples))}\n")
        f.write("="*80 + "\n\n")
        
        for i, ex in enumerate(tp_examples[:25], 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"ID: {ex['id']}\n")
            f.write(f"Text: {ex['text'][:200]}...\n")
            f.write(f"Pronoun: '{ex['pronoun']}'\n")
            f.write(f"True Answer: '{ex['true_answer']}'\n")
            f.write(f"Predicted: '{ex['predicted']}' (Score: {ex['score']:.4f})\n")
            f.write(f"Result: ✓ CORRECT\n\n")
        
        # False Positives
        f.write("\n" + "="*80 + "\n")
        f.write("FALSE POSITIVES (FP): Incorrectly Predicted as Antecedent\n")
        f.write(f"Total: {len(fp_examples)}, Showing: {min(25, len(fp_examples))}\n")
        f.write("="*80 + "\n\n")
        
        for i, ex in enumerate(fp_examples[:25], 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"ID: {ex['id']}\n")
            f.write(f"Text: {ex['text'][:200]}...\n")
            f.write(f"Pronoun: '{ex['pronoun']}'\n")
            f.write(f"True Answer: '{ex['true_answer']}'\n")
            f.write(f"Predicted: '{ex['predicted']}' (Score: {ex['score']:.4f})\n")
            f.write(f"Result: ✗ WRONG (Predicted wrong candidate)\n\n")
        
        # False Negatives
        f.write("\n" + "="*80 + "\n")
        f.write("FALSE NEGATIVES (FN): Failed to Predict Correct Antecedent\n")
        f.write(f"Total: {len(fn_examples)}, Showing: {min(25, len(fn_examples))}\n")
        f.write("="*80 + "\n\n")
        
        for i, ex in enumerate(fn_examples[:25], 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"ID: {ex['id']}\n")
            f.write(f"Text: {ex['text'][:200]}...\n")
            f.write(f"Pronoun: '{ex['pronoun']}'\n")
            f.write(f"True Answer: '{ex['true_answer']}'\n")
            if ex.get('predicted'):
                f.write(f"Predicted: '{ex['predicted']}' (Score: {ex.get('score', 'N/A'):.4f})\n")
            else:
                f.write(f"Predicted: None ({ex.get('reason', 'No prediction made')})\n")
            f.write(f"Result: ✗ WRONG (Failed to find correct answer)\n\n")
        
        # True Negatives
        f.write("\n" + "="*80 + "\n")
        f.write("TRUE NEGATIVES (TN): Correctly Rejected as Non-Antecedent\n")
        f.write(f"Total: {len(tn_examples)}, Showing: {min(25, len(tn_examples))}\n")
        f.write("="*80 + "\n\n")
        
        if len(tn_examples) == 0:
            f.write("No True Negative cases found.\n")
            f.write("\nWHY NO TRUE NEGATIVES?\n")
            f.write("-" * 80 + "\n")
            f.write("In this coreference resolution task, we ALWAYS make exactly one prediction\n")
            f.write("per example (selecting the best-scoring candidate). We never predict 'NONE'.\n")
            f.write("Therefore:\n")
            f.write("  - A True Negative would require: predicting 'no antecedent' when there\n")
            f.write("    actually is no antecedent.\n")
            f.write("  - Since we always predict a candidate, we can never have TN = 0.\n")
            f.write("  - Every prediction is either TP (correct) or FP (wrong candidate chosen).\n")
            f.write("  - Every case is either TP (correct) or FN (missed correct candidate).\n\n")
        else:
            for i, ex in enumerate(tn_examples[:25], 1):
                f.write(f"--- Example {i} ---\n")
                f.write(f"ID: {ex['id']}\n")
                f.write(f"Text: {ex['text'][:200]}...\n\n")
    
    print(f"\nExamples saved to 'nb_examples_detailed.txt'")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    train_rows = load_gap_dataset("train.csv")
    test_rows  = load_gap_dataset("test.csv")

    X_train, y_train, meta_train = prepare_data(train_rows)
    print(f"Class distribution: {dict(Counter(y_train))}")

    model = train_nb(X_train, y_train)
    evaluate_nb(model, test_rows, save_examples=True)

