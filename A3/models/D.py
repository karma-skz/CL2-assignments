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

FUNCTION_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",
    'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers',
    'herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if',
    'or','because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out',
    'on','off','over','under','again','further','then','once','here','there','when','where','why',
    'how','all','any','both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",
    'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',
    "couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',
    "haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",
    'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",
    'wouldn',"wouldn't"
}

WORD_RE = re.compile(r"\w+|[.!?;,]")
def tokenize_content(text):
    toks = WORD_RE.findall(clean(text).lower())
    words = [t for t in toks if re.match(r"^\w+$", t)]
    return [w for w in words if w not in FUNCTION_WORDS]

# ==========================================================================
def train(train_neg, train_pos):
    neg_wc = defaultdict(int)
    pos_wc = defaultdict(int)
    neg_total = pos_total = 0
    
    for l in train_neg:
        for w in tokenize_content(l):
            neg_wc[w] += 1
            neg_total += 1
    for l in train_pos:
        for w in tokenize_content(l):
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
    words = tokenize_content(text)
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
print("iv) acc, prec, rec, f1 ->", tuple(round(x*100,4) for x in (acc, prec, rec, f1)))
print("iv) confusion matrix (tp, tn, fp, fn) ->", (tp, tn, fp, fn))
