import re
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import math

path = Path(__file__).resolve().parent
dataset_1 = path / "pg10.txt"
dataset_2 = path / "pg35997.txt"

# read + lowercase
with open(dataset_1, "r", encoding="utf-8", errors="ignore") as f:
    raw_data = f.read()
with open(dataset_2, "r", encoding="utf-8", errors="ignore") as f:
    raw_data += f.read()
raw_data = raw_data.lower()

words = re.findall(r'[a-zA-Z]+', raw_data) # extract words (a word = one or more a–zA-Z characters)

word_freq = Counter(words) 
lengths = [len(word) for word in words] # measure length of each word

freq = Counter(lengths) # count frequency of each length

lengths_sorted = sorted(freq.items()) # sort by length

# measure length – done above
# number of words at different lengths
print("Word length\tFrequency")
for length, count in lengths_sorted:
    print(f"{length}\t\t{count}")

# shortest words in the dataset
min_len = min(freq.keys())
sorted_by_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
print(f"\nShortest length: {min_len}")
print("Shortest words in the dataset:")
shortest_words = [word for word, count in sorted_by_freq if len(word) == min_len]
for word in shortest_words:
    print(f"{word}\t{word_freq[word]}")

print()
for i in range(50):
    print(sorted_by_freq[i])

# plot length vs frequency
x = [length for length, count in lengths_sorted]
y = [count for length, count in lengths_sorted]

plt.figure()
plt.plot(x, y, marker='o')
plt.xlabel('Word length')
plt.ylabel('Frequency')
plt.title('Word length vs Frequency')
plt.show()

# plot log10(length) vs log10(frequency)
log_x = np.log10(x)
log_y = np.log10(y)

plt.figure()
plt.plot(log_x, log_y, marker='o')
plt.xlabel('log10(Word length)')
plt.ylabel('log10(Frequency)')
plt.title('Log-Log Plot of Word length vs Frequency')
plt.show()

def pearson(a, b):
    n = len(a)
    mean_a = sum(a)/n
    mean_b = sum(b)/n

    # covariance of x and y / var of x * var of y
    num = sum((a[i]-mean_a) * (b[i]-mean_b) for i in range(n))
    den = math.sqrt(sum((a[i]-mean_a)**2 for i in range(n)) *
                    sum((b[i]-mean_b)**2 for i in range(n)))
    return num/den

pearson_coeff = pearson(log_x, log_y)
print("\n\npearson coefficient:", pearson_coeff)