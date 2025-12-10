# Naive Bayes Classifier Results

## Confusion Matrices

### i) Bag of words (exclude unseen)  

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------| 
| Actual Positive | 1226 (TP)         | 374 (FN)           |
| Actual Negative | 356 (FP)          | 1244 (TN)          |


### ii) Bag of words (include unseen)  

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | 1223 (TP)         | 377 (FN)           |
| Actual Negative | 355 (FP)          | 1245 (TN)          |


### iii) Bag of words (binarization)  

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | 1232 (TP)         | 368 (FN)           |
| Actual Negative | 373 (FP)          | 1227 (TN)          |


### iv) Content words (frequencies)  

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | 1232 (TP)         | 368 (FN)           |
| Actual Negative | 398 (FP)          | 1202 (TN)          |


### v) Content words (binarization)  

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | 1202 (TP)         | 398 (FN)           |
| Actual Negative | 348 (FP)          | 1252 (TN)          |


### vi) Negation features (frequencies)  

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | 1233 (TP)         | 367 (FN)           |
| Actual Negative | 384 (FP)          | 1216 (TN)          |


### vii) Negation features (binarization)  

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | 1235 (TP)         | 365 (FN)           |
| Actual Negative | 372 (FP)          | 1228 (TN)          |


## Performance Metrics

| Classifier                           | Accuracy | Precision | Recall  | F1-Score |
|--------------------------------------|----------|-----------|---------|----------|
| i) Bag of words (exclude unseen)     | 77.19    | 77.50     | 76.63   | 77.06    |
| ii) Bag of words (include unseen)    | 77.13    | 77.50     | 76.44   | 76.97    |
| iii) Bag of words (binarization)     | 76.84    | 76.76     | 77.00   | 76.88    |
| iv) Content words (frequencies)      | 76.06    | 75.58     | 77.00   | 76.28    |
| v) Content words (binarization)      | 76.69    | 77.55     | 75.13   | 76.32    |
| vi) Negation features (frequencies)  | 76.53    | 76.25     | 77.06   | 76.66    |
| vii) Negation features (binarization)| 76.97    | 76.85     | 77.19   | 77.02    |

---
## Misclassification Analysis

### Model i) Bag of words (exclude unseen)

**False Negatives:**  
- "neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway ."  
- "i won't argue with anyone who calls 'slackers' dumb , insulting , or childish . . . but i laughed so much that i didn't mind ."  

**False Positives:**  
- "mr . wollter and ms . seldhal give strong and convincing performances , but neither reaches into the deepest recesses of the character to unearth the quaking essence of passion , grief and fear ."  
- "the makers of divine secrets of the ya-ya sisterhood should offer a free ticket ( second prize , of course , two free tickets ) to anyone who can locate a genuinely honest moment in their movie ."  

**Linguistic Analysis:**  
The basic bag-of-words model struggles with complex syntactic structures that reverse sentiment polarity. It misses contrastive conjunctions (*“but,” “nevertheless”*) that signal sentiment shifts and ironic constructions where positive words appear in negative contexts. Word frequencies are treated equally, without understanding syntactic relationships or discourse structure.  

---

### Model ii) Bag of words (include unseen)

**False Negatives:**  
- "neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway ."  
- "opening with some contrived banter , cliches and some loose ends , the screenplay only comes into its own in the second half ."  

**False Positives:**  
- "mr . wollter and ms . seldhal give strong and convincing performances , but neither reaches into the deepest recesses of the character to unearth the quaking essence of passion , grief and fear ."  
- "jacquot's rendering of puccini's tale of devotion and double-cross is more than just a filmed opera . in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically ."  

**Linguistic Analysis:**  
This approach fails with hedged language (*“only,” “slightly”*) and qualified praise/criticism. Allowing unseen words does not address the core issue of semantic scope — positive words like *“strong,” “convincing”* are weighted equally regardless of their modification by negation or contrast markers.  

---

### Model iii) Bag of words (binarization)

**False Negatives:**  
- "neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway ."  
- "it's plotless , shapeless -- and yet , it must be admitted , not entirely humorless . indeed , the more outrageous bits achieve a shock-you-into-laughter intensity of almost dadaist proportions ."  

**False Positives:**  
- "mr . wollter and ms . seldhal give strong and convincing performances , but neither reaches into the deepest recesses of the character to unearth the quaking essence of passion , grief and fear ."  
- "a film that will be best appreciated by those willing to endure its extremely languorous rhythms , waiting for happiness is ultimately thoughtful without having much dramatic impact ."  

**Linguistic Analysis:**  
Binarization collapses word frequency into a simple 0/1 presence indicator. This means the model cannot distinguish between emphasis through repetition (*“good good good”* vs. *“good”*) or the distribution of sentiment-laden words across a review. While intensity markers such as *“extremely”* or *“ultimately”* are still registered if they occur once, their relative weight is treated the same as any other word. As a result, the model misses gradience in sentiment strength and continues to treat positive and negative tokens without regard to their syntactic roles or discourse context.  

---

### Model iv) Content words (frequencies)

**False Negatives:**  
- "neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway ."  
- "opening with some contrived banter , cliches and some loose ends , the screenplay only comes into its own in the second half ."  

**False Positives:**  
- "jacquot's rendering of puccini's tale of devotion and double-cross is more than just a filmed opera . in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically ."  
- "these are textbook lives of quiet desperation ."  

**Linguistic Analysis:**  
Removing function words eliminates critical sentiment connectives (*“but,” “however,” “nevertheless”*) that signal argumentative structure. Content words like *“amusing,” “devotion”* suggest positive sentiment, but the model misses discourse markers that indicate these words appear in negative contexts or qualified statements.  

---

### Model v) Content words (binarization)

**False Negatives:**  
- "neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway ."  
- "on the surface a silly comedy , scotland , pa would be forgettable if it weren't such a clever adaptation of the bard's tragic play ."  

**False Positives:**  
- "mr . wollter and ms . seldhal give strong and convincing performances , but neither reaches into the deepest recesses of the character to unearth the quaking essence of passion , grief and fear ."  
- "though it was made with careful attention to detail and is well-acted by james spader and maggie gyllenhaal , i felt disrespected ."  

**Linguistic Analysis:**  
This model combines the weaknesses of both content-word filtering and binarization. Conditional structures (*“if it weren't”*) and concessive clauses (*“though...”*) are stripped of their logical operators, leaving only content words that mislead about overall sentiment. The model loses both gradience from word frequency and logical relationships from function words.  

---

### Model vi) Negation features (frequencies)

**False Negatives:**  
- "neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway ."  
- "i don't feel the least bit ashamed in admitting that my enjoyment came at the expense of seeing justice served , even if it's a dish that's best served cold ."  

**False Positives:**  
- "mr . wollter and ms . seldhal give strong and convincing performances , but neither reaches into the deepest recesses of the character to unearth the quaking essence of passion , grief and fear ."  
- "jacquot's rendering of puccini's tale of devotion and double-cross is more than just a filmed opera . in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically ."  

**Linguistic Analysis:**  
Simple negation scope rules fail with complex coordinated negation (*“neither…nor”*) and multiple negations within a single sentence. The scope of negation often extends beyond single words to entire clauses, which punctuation-based rules miss. Additionally, descriptive negation (*“neither reaches into deepest recesses”*) differs from evaluative negation but is treated identically.  

---

### Model vii) Negation features (binarization)

**False Negatives:**  
- "neither the funniest film that eddie murphy nor robert de niro has ever made , showtime is nevertheless efficiently amusing for a good while . before it collapses into exactly the kind of buddy cop comedy it set out to lampoon , anyway ."  
- "remember the kind of movie we were hoping 'ecks vs . sever' or 'xxx' was going to be ? this is it ."  

**False Positives:**  
- "mr . wollter and ms . seldhal give strong and convincing performances , but neither reaches into the deepest recesses of the character to unearth the quaking essence of passion , grief and fear ."  
- "a film that will be best appreciated by those willing to endure its extremely languorous rhythms , waiting for happiness is ultimately thoughtful without having much dramatic impact ."  

**Linguistic Analysis:**  
This model inherits the scope problems of Model vi while also losing gradience through binarization. Rhetorical questions and implicit comparisons (*“this is it” = positive evaluation*) require pragmatic inference beyond simple negation handling. The model struggles with inferential sentiment where positivity emerges from context rather than explicit sentiment words.  

---

## Key Linguistic Patterns Across All Models

- **Syntactic Complexity:** All models fail with coordination, subordination, and embedded clauses.  
- **Discourse Markers:** Critical sentiment-shifting words (*“but,” “however,” “nevertheless”*) are ignored or inadequately weighted.  
- **Scope Issues:** Negation, qualification, and modification scopes extend beyond word-level boundaries.  
- **Pragmatic Inference:** Subtle rhetorical strategies (irony, understatement, implicit comparison) require world knowledge.  
- **Register Sensitivity:** Formal review language relies on hedging and qualification, which differ from casual sentiment expression.  

**Overall:** Despite different feature representations, all models perform similarly poorly on reviews with complex discourse and rhetorical structures.


# Opinion Lexicon Enhancement

## Implementation
A polarity lexicon approach was implemented using the **Opinion Lexicon of Hu and Liu** (2004), which contains approximately 6,800 positive and negative opinion words compiled from customer reviews. This lexicon is particularly suitable for sentiment analysis as it focuses on subjective, evaluative terms commonly used in opinion expressions rather than general sentiment words.

## Expected vs. Actual Performance
**Expected:** The lexicon-based approach should improve precision by focusing on known sentiment-bearing words, potentially reducing false positives from neutral descriptive language. However, recall might suffer from the lexicon's limited coverage of domain-specific terminology.

**Actual Results:**
- **Accuracy:** 77.91% (↑0.72% from baseline Model i: 77.19%)
- **Precision:** 77.82% (↑0.32% from baseline: 77.50%)
- **Recall:** 78.06% (↑1.43% from baseline: 76.63%)
- **F1-Score:** 77.94% (↑0.88% from baseline: 77.06%)

## Analysis
The Opinion Lexicon approach achieved modest but consistent improvements across all metrics, with recall showing the most significant gain. This suggests the lexicon successfully captured additional positive sentiment expressions missed by the basic bag-of-words model. The balanced improvement in both precision and recall indicates that the curated sentiment vocabulary reduced both false negatives (missed positive reviews) and false positives (incorrectly classified negative reviews), demonstrating the value of domain-specific sentiment knowledge over purely statistical word frequency approaches.