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
def tokenize_words(text):
    text = clean(text)
    toks = WORD_RE.findall(text.lower())
    return [t for t in toks if re.match(r"^\w+$", t)]

# ==========================================================================
def load_opinion_lexicon(pos_file="list_pos.txt", neg_file="list_neg.txt"):
    with open(pos_file, "r", errors="ignore") as f:
        pos_lex = {w.strip() for w in f if w.strip() and not w.startswith(";")}
    with open(neg_file, "r", errors="ignore") as f:
        neg_lex = {w.strip() for w in f if w.strip() and not w.startswith(";")}
    return pos_lex, neg_lex

pos_lex, neg_lex = load_opinion_lexicon()

# ==========================================================================
def train(train_neg, train_pos):
    neg_wc = defaultdict(int)
    pos_wc = defaultdict(int)
    neg_total = pos_total = 0
    
    for line in train_neg:
        for w in tokenize_words(line):
            neg_wc[w] += 1
            neg_total += 1
    for line in train_pos:
        for w in tokenize_words(line):
            pos_wc[w] += 1
            pos_total += 1
    
    vocab = set(neg_wc.keys()) | set(pos_wc.keys())
    vocab_size = len(vocab)
    p_neg = len(train_neg) / (len(train_neg) + len(train_pos))
    p_pos = len(train_pos) / (len(train_neg) + len(train_pos))
    
    return {
        'neg_wc': neg_wc, 'pos_wc': pos_wc,
        'neg_total': neg_total, 'pos_total': pos_total,
        'vocab': vocab, 'vocab_size': vocab_size,
        'p_neg': p_neg, 'p_pos': p_pos
    }

def predict(text, model, pos_lex, neg_lex):
    words = tokenize_words(text)
    neg_wc = model['neg_wc']; pos_wc = model['pos_wc']
    neg_total = model['neg_total']; pos_total = model['pos_total']
    V = model['vocab_size']
    
    log_neg = math.log(model['p_neg'])
    log_pos = math.log(model['p_pos'])
    
    for w in words:
        if w not in model['vocab']:
            continue
        p_w_neg = (neg_wc.get(w,0) + 1) / (neg_total + V)
        p_w_pos = (pos_wc.get(w,0) + 1) / (pos_total + V)
        log_neg += math.log(p_w_neg)
        log_pos += math.log(p_w_pos)
    
    # --- Lexicon-based features ---
    pos_hits = sum(1 for w in words if w in pos_lex)
    neg_hits = sum(1 for w in words if w in neg_lex)
    log_pos += pos_hits * 0.8   # tweakable weight
    log_neg += neg_hits * 0.8
    
    return 'neg' if log_neg > log_pos else 'pos'

def evaluate(model, neg_val, pos_val, pos_lex, neg_lex):
    tp = tn = fp = fn = 0
    for l in pos_val:
        pred = predict(l, model, pos_lex, neg_lex)
        if pred == 'pos': tp += 1
        else: fn += 1
    for l in neg_val:
        pred = predict(l, model, pos_lex, neg_lex)
        if pred == 'neg': tn += 1
        else: fp += 1
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1, tp, tn, fp, fn

# ==========================================================================
model = train(neg_train, pos_train)
acc, prec, rec, f1, tp, tn, fp, fn = evaluate(model, neg_test, pos_test, pos_lex, neg_lex)

print("With Opinion Lexicon features:")
print("acc, prec, rec, f1 ->", tuple(round(x*100,4) for x in (acc, prec, rec, f1)))
print("confusion matrix (tp, tn, fp, fn) ->", (tp, tn, fp, fn))
