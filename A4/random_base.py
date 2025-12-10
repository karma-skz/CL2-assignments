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
        if hasattr(subtree, 'label') and subtree.label() == 'NP': # type: ignore (only for vs code apparently, else its just a warning)
            np = ' '.join([w for w, t in subtree.leaves()]) # type: ignore
            nps.append(np)
    return nps

def random_eval_once(data):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for row in data:
        nps = extract_candidates(row["Text"], int(row["Pronoun-offset"]))
        if not nps:
            if row['A-coref'] == 'True' or row['B-coref'] == 'True':
                fn += 1
            else:
                tn += 1
            continue

        chosen_np = random.choice(nps)
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

# run 100 times and get average
def random_eval_average(data):
    accs, precs, recs, f1s = [], [], [], []
    for _ in range(100):
        acc, prec, rec, f1 = random_eval_once(data)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    
    avg_acc = sum(accs) / len(accs)
    avg_prec = sum(precs) / len(precs)
    avg_rec = sum(recs) / len(recs)
    avg_f1 = sum(f1s) / len(f1s)
    
    print(f"Random Baseline (100 runs):")
    print(f"  Average Accuracy:  {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"  Average Precision: {avg_prec:.4f} ({avg_prec*100:.2f}%)")
    print(f"  Average Recall:    {avg_rec:.4f} ({avg_rec*100:.2f}%)")
    print(f"  Average F1-Score:  {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print(f"\n  Accuracy  - Min: {min(accs):.4f}, Max: {max(accs):.4f}")
    print(f"  Precision - Min: {min(precs):.4f}, Max: {max(precs):.4f}")
    print(f"  Recall    - Min: {min(recs):.4f}, Max: {max(recs):.4f}")
    print(f"  F1-Score  - Min: {min(f1s):.4f}, Max: {max(f1s):.4f}")
    return avg_acc, avg_prec, avg_rec, avg_f1

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

    # run random baseline
    random_eval_average(data)
