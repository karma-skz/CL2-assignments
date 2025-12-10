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
NEGATION_TOKENS = {"not", "n't", "no", "never", "cant", "don't", "didn't", "didnt", "can't", "cannot"}

def tokenize_with_negation(text):
    """Apply negation feature: prepend NOT to words after negation tokens until punctuation"""
    toks = WORD_RE.findall(clean(text).lower())
    res = []
    negating = False
    
    for t in toks:
        if t in {'.', '!', '?', ';', ','}:
            negating = False
            res.append(t)
            continue
        if t in NEGATION_TOKENS:
            negating = True
            res.append(t)
            continue
        if negating and re.match(r"^\w+$", t):
            res.append("NOT")
            res.append(t)
        else:
            res.append(t)
    return [x for x in res if re.match(r"^\w+$", x)]

# ==========================================================================
def train(train_neg, train_pos):
    neg_wc = defaultdict(int)
    pos_wc = defaultdict(int)
    neg_total = pos_total = 0
    
    for l in train_neg:
        for w in tokenize_with_negation(l):
            neg_wc[w] += 1
            neg_total += 1
    for l in train_pos:
        for w in tokenize_with_negation(l):
            pos_wc[w] += 1
            pos_total += 1
    
    vocab = set(neg_wc.keys()) | set(pos_wc.keys())
    p_neg = len(train_neg) / (len(train_neg) + len(train_pos))
    p_pos = len(train_pos) / (len(train_neg) + len(train_pos))
    
    return {
        'neg_wc': neg_wc, 'pos_wc': pos_wc,
        'neg_total': neg_total, 'pos_total': pos_total,
        'vocab': vocab, 'p_neg': p_neg, 'p_pos': p_pos
    }

def predict(text, model):
    words = tokenize_with_negation(text)
    neg_wc = model['neg_wc']; pos_wc = model['pos_wc']
    neg_total = model['neg_total']; pos_total = model['pos_total']
    V = len(model['vocab'])
    
    log_neg = math.log(model['p_neg'])
    log_pos = math.log(model['p_pos'])
    
    for w in words:
        if w not in model['vocab']:
            continue
        p_w_neg = (neg_wc.get(w, 0) + 1) / (neg_total + V)
        p_w_pos = (pos_wc.get(w, 0) + 1) / (pos_total + V)
        log_neg += math.log(p_w_neg)
        log_pos += math.log(p_w_pos)
    
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
print("vi) acc, prec, rec, f1 ->", tuple(round(x*100,4) for x in (acc, prec, rec, f1)))
print("vi) confusion matrix (tp, tn, fp, fn) ->", (tp, tn, fp, fn))
