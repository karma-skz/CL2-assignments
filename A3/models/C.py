import random, re, math
from collections import defaultdict
from cleaner import clean
random.seed(0)

with open("neg.txt", "r", errors="ignore") as f:
    neg = f.readlines()
with open("pos.txt", "r", errors="ignore") as f:
    pos = f.readlines()

def split_data(data):
    random.shuffle(data)
    n = len(data)
    return data[:int(0.7*n)], data[int(0.7*n):]

neg_train, neg_test = split_data(neg)
pos_train, pos_test = split_data(pos)

WORD_RE = re.compile(r"\w+|[.!?;,]")
def doc_words(text):
    toks = WORD_RE.findall(clean(text).lower())
    words = [t for t in toks if re.match(r"^\w+$", t)]
    return set(words)

# ==========================================================================
def train(train_neg, train_pos):
    neg_doc_counts = defaultdict(int)
    pos_doc_counts = defaultdict(int)
    
    for l in train_neg:
        for w in doc_words(l):
            neg_doc_counts[w] += 1
    for l in train_pos:
        for w in doc_words(l):
            pos_doc_counts[w] += 1
    
    vocab = set(neg_doc_counts.keys()) | set(pos_doc_counts.keys())
    n_neg = len(train_neg)
    n_pos = len(train_pos)
    p_neg = n_neg / (n_neg + n_pos)
    p_pos = n_pos / (n_neg + n_pos)
    
    return {
        'neg_doc_counts': neg_doc_counts, 'pos_doc_counts': pos_doc_counts,
        'vocab': vocab, 'n_neg': n_neg, 'n_pos': n_pos,
        'p_neg': p_neg, 'p_pos': p_pos
    }

def predict(text, model):
    words = doc_words(text)
    vocab = model['vocab']
    
    log_neg = math.log(model['p_neg'])
    log_pos = math.log(model['p_pos'])
    
    # Bernoulli Naive Bayes: for each word in vocab, calculate P(word|class) and P(!word|class)
    for w in vocab:
        # Laplace smoothing on document-level counts
        p_w_neg = (model['neg_doc_counts'].get(w, 0) + 1) / (model['n_neg'] + 2)
        p_w_pos = (model['pos_doc_counts'].get(w, 0) + 1) / (model['n_pos'] + 2)
        
        if w in words:  # Word is present in document
            log_neg += math.log(p_w_neg)
            log_pos += math.log(p_w_pos)
        else:  # Word is absent from document
            log_neg += math.log(1 - p_w_neg)
            log_pos += math.log(1 - p_w_pos)
    
    return 'neg' if log_neg > log_pos else 'pos'

def evaluate(model, neg_val, pos_val):
    tp = tn = fp = fn = 0
    for l in pos_val:
        pred = predict(l, model)
        if pred == 'pos': tp += 1
        else: fn += 1
    for l in neg_val:
        pred = predict(l, model)
        if pred == 'neg': tn += 1
        else: fp += 1
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1, tp, tn, fp, fn

# ==========================================================================
model = train(neg_train, pos_train)
acc, prec, rec, f1, tp, tn, fp, fn = evaluate(model, neg_test, pos_test)
print("iii) acc, prec, rec, f1 ->", tuple(round(x*100,4) for x in (acc, prec, rec, f1)))
print("iii) confusion matrix (tp, tn, fp, fn) ->", (tp, tn, fp, fn))
