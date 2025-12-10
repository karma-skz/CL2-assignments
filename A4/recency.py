import csv
import random
import nltk
from nltk import word_tokenize, pos_tag, RegexpParser

random.seed(0)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def load_data(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def extract_candidates(text, pronoun_offset):
    text_before = text[:pronoun_offset]
    tokens = word_tokenize(text_before)
    tagged = pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    parser = RegexpParser(grammar)
    tree = parser.parse(tagged)

    nps = []
    for subtree in tree:
        if hasattr(subtree, 'label') and subtree.label() == 'NP': # type: ignore
            np = ' '.join([w for w, t in subtree.leaves()]) # type: ignore
            nps.append(np)
    return nps

def eval_once(data):
    tp, fp, tn, fn = 0, 0, 0, 0
    for row in data:
        nps = extract_candidates(row["Text"], int(row["Pronoun-offset"]))
        if not nps:
            if row['A-coref'] == 'True' or row['B-coref'] == 'True':
                fn += 1
            else:
                tn += 1
            continue

        chosen_np = nps[-1]
        correct_antecedent = row['A'] if row['A-coref'] == 'True' else row['B']
        
        if chosen_np == correct_antecedent:
            tp += 1
        else:
            fp += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

# evaluate recency baseline
def recency_eval(data):
    acc, prec, rec, f1 = eval_once(data)
    print(f"Recency Baseline:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    return acc, prec, rec, f1

if __name__ == "__main__":
    data = load_data("test.csv")
    
    # show candidate extraction example
    # sample = data[0]
    # pron_offset = int(sample["Pronoun-offset"])
    # nps = extract_nouns_before_pronoun(sample["Text"], pron_offset)
    # print(f"Example - Candidate Extraction:")
    # print(f"  Pronoun: '{sample['Pronoun']}'")
    # print(f"  Extracted {len(nps)} noun phrases")
    # print(f"  Last 3: {[np for np, dist in nps[-3:]]}\n")

    # run recency baseline
    recency_eval(data)
