#!/usr/bin/env python3

STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
"you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
"couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
"haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
'wouldn', "wouldn't"}

def tokenize(text):
    tokens = text.lower().replace(',', ' ').replace('.', ' ').replace(':', ' ').replace(';', ' ').split()
    return [t.strip('.,;:!?"\'()') for t in tokens if t.strip('.,;:!?"\'()') and t.lower() not in STOPWORDS]

def calculate_metrics(reference, candidate):
    """Calculate precision, recall, and F1 score"""
    ref_tokens = set(tokenize(reference))
    cand_tokens = set(tokenize(candidate))
    
    if not cand_tokens:
        return 0.0, 0.0, 0.0
    
    overlap = ref_tokens & cand_tokens
    
    precision = len(overlap) / len(cand_tokens) if cand_tokens else 0.0
    recall = len(overlap) / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def parse_report(filename='rst_summarization_report.txt'):
    """Extract summaries from report file"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('PARAGRAPH'):
            para_num = lines[i].strip().split()[1]
            summary = lines[i+2].replace('Summary: ', '').strip()
            llm = lines[i+3].replace('LLM: ', '').strip()
            data.append({'para': para_num, 'summary': summary, 'llm': llm})
            i += 4
        else:
            i += 1
    
    return data

def main():
    """Calculate and report metrics"""
    data = parse_report()
    
    results = []
    for item in data:
        prec, rec, f1 = calculate_metrics(item['llm'], item['summary'])
        results.append({'para': item['para'], 'precision': prec, 'recall': rec, 'f1': f1})
    
    avg_prec = sum(r['precision'] for r in results) / len(results)
    avg_rec = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    
    with open('metrics_report.txt', 'w', encoding='utf-8') as f:
        f.write("EVALUATION METRICS: Rule-based vs LLM Summaries\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"Paragraph {r['para']}: P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1']:.3f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"AVERAGE: P={avg_prec:.3f}, R={avg_rec:.3f}, F1={avg_f1:.3f}\n")

if __name__ == '__main__':
    main()
