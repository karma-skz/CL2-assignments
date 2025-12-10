import csv
import math
import unicodedata
from collections import Counter

def normalize(text):
	# t = unicodedata.normalize("NFC", text).lower()
	t=text.lower()
	out = []
	prev_space = False
	for ch in t:
		if ch.isalpha():
			out.append(ch)
			prev_space = False
		elif ch.isspace():
			if not prev_space:
				out.append(" ")
			prev_space = True
	return "".join(out).strip()

def char_ngrams(text, n):
    if n <= 0:
        return []
    text = text or ""
    if len(text) < n:
        return []
    grams = []
    for i in range(len(text) - n + 1):
        grams.append(text[i : i + n])
    return grams

def compute_counts(texts, n):
    c = Counter()
    for s in texts:
        for g in char_ngrams(s, n):
            c[g] += 1
    return c

def read_training(path):
    ces_texts = []
    pol_texts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lan = (row.get("lan_code") or "").strip()
            sent = row.get("sentence") or ""
            if not sent:
                continue
            ns = normalize(sent)
            if not ns:
                continue
            if lan == "ces":
                ces_texts.append(ns)
            elif lan == "pol":
                pol_texts.append(ns)
    return ces_texts, pol_texts

def read_test(path):
	rows = []
	with open(path, "r", encoding="utf-8") as f:
		for row in csv.DictReader(f):
			lan = (row.get("lan_code") or "").strip()
			sent = row.get("sentence") or ""
			if not sent:
				continue
			ns = normalize(sent)
			if not ns:
				continue
			rows.append((lan, ns))
	return rows


def predict(sentence, models, n):
	grams = char_ngrams(sentence, n)
	if not grams:
		return max(models.items(), key=lambda kv: sum(kv[1].values()))[0]

	best_lang = None
	best_score = -1
	for lan in sorted(models.keys()):
		cnt = models[lan]
		score = 0
		for g in grams:
			score += cnt.get(g, 0)
		if score > best_score:
			best_lang, best_score = lan, score
	return best_lang or sorted(models.keys())[0]


def evaluate(test_rows, models, n):
	langs = sorted(models.keys())
	conf = {t: {p: 0 for p in langs} for t in langs}
	for true_lan, sent in test_rows:
		if true_lan not in models:
			continue
		pred = predict(sent, models, n)
		conf[true_lan][pred] += 1
	return conf


def per_class_metrics(conf):
	langs = sorted(conf.keys())
	res = {}
	for lan in langs:
		tp = conf[lan].get(lan, 0)
		pred_pos = sum(conf[t][lan] for t in langs)
		true_pos = sum(conf[lan].values())
		p = tp / pred_pos if pred_pos else 0.0
		r = tp / true_pos if true_pos else 0.0
		f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
		res[lan] = (p, r, f1)
	return res


def macro_avg(metrics):
	n = len(metrics) or 1
	mp = sum(p for p, _, _ in metrics.values()) / n
	mr = sum(r for _, r, _ in metrics.values()) / n
	mf = sum(f for _, _, f in metrics.values()) / n
	return mp, mr, mf


def accuracy(conf):
	correct = sum(conf[l].get(l, 0) for l in conf)
	total = sum(sum(conf[l].values()) for l in conf)
	return (correct / total) if total else 0.0, correct, total

ces_texts, pol_texts = read_training("Language_datasets/polish_czech_train.csv")
test_rows = read_test("Language_datasets/polish_czech_test.csv")

lines = [
    "# Results: Polish vs Czech (character n-grams)",
    "",
    f"- Train file: Language_datasets/polish_czech_train.csv",
    f"- Test file: Language_datasets/polish_czech_test.csv",
    "",
]

for n in (1, 2, 3, 4):
    models = {
        "ces": compute_counts(ces_texts, n),
        "pol": compute_counts(pol_texts, n),
    }
    conf = evaluate(test_rows, models, n)
    cls = per_class_metrics(conf)
    mp, mr, mf = macro_avg(cls)
    acc, correct, total = accuracy(conf)

    lines += [
        f"## {n}-gram",
        f"- Accuracy: {acc:.6f} ({correct}/{total})",
        f"- Macro Precision: {mp:.6f}",
        f"- Macro Recall: {mr:.6f}",
        f"- Macro F1: {mf:.6f}",
        "",
        "| Language | Precision | Recall | F1 |",
        "| --- | --- | --- | --- |",
        f"| ces | {cls['ces'][0]:.6f} | {cls['ces'][1]:.6f} | {cls['ces'][2]:.6f} |",
        f"| pol | {cls['pol'][0]:.6f} | {cls['pol'][1]:.6f} | {cls['pol'][2]:.6f} |",
        "",
    ]

with open("result_pol_ces.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Wrote report for polish vs czech")
