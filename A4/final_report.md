# Anaphora Resolution - Implementation Report

## Overview

This report presents three different approaches to anaphora resolution on the GAP (Gendered Ambiguous Pronouns) dataset:
1. **Random Baseline** - Random selection among candidates
2. **Recency Baseline** - Most recent noun phrase selection
3. **Naive Bayes Classifier** - Feature-based approach

---

## 1. Random Baseline

### Approach
The random baseline extracts all noun phrase candidates appearing before the pronoun using a simple grammar pattern (`NP: {<DT>?<JJ.*>*<NN.*>+}`). From these candidates, it randomly selects one as the predicted antecedent.

### Implementation Details
- **Candidate Extraction**: Uses NLTK's RegexpParser with POS tags to identify noun phrases
- **Selection Strategy**: Random choice using `random.choice()`
- **Evaluation**: Averaged over 100 runs to account for randomness

### Results

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| **Accuracy** | 7.68% | 6.15% | 8.75% |
| **Precision** | 7.68% | 6.15% | 8.75% |
| **Recall** | 99.35% | 99.19% | 99.43% |
| **F1-Score** | 14.26% | 11.59% | 16.09% |

### Analysis
- Very low precision (7.68%) shows predictions are mostly incorrect
- This serves as a lower bound for comparison which means that anything that is based on training a model should at least perform better than this.

---

## 2. Recency Baseline

### Approach
The recency baseline extracts noun phrase candidates and selects the last (most recent) one before the pronoun.

### Implementation Details
- **Candidate Extraction**: Same grammar-based approach as random baseline
- **Selection Strategy**: Always selects `nps[-1]` (most recent noun phrase)

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 11.10% |
| **Precision** | 11.11% |
| **Recall** | 99.55% |
| **F1-Score** | 19.98% |

### Analysis
- Slight improvement over random baseline (11.10% vs 7.68% accuracy)
- Maintains very high recall (99.55%)
- Shows that recency is a meaningful but insufficient heuristic

---

   ## 3. Naive Bayes (short summary)

   Features implemented
   - token_distance (exact offset) and distance_bin (close / medium / far)
   - same_sentence and a boost_same_sentence flag for tie-breaking
   - pronoun gender, number_agree and a simple gender_agree heuristic
   - boost_close_distance and head_match

   Fallback / tie behavior
   - Candidates are ranked by the model's log-score. If scores are effectively tied, we prefer candidates in the same sentence and ones that are very close to the pronoun.

   Metrics
   - Accuracy 16.86 · Precision 16.87 · Recall 16.86 · F1 16.87

   Why TN = 0
   - The system always picks one candidate (no "none" option) and every test case has a labelled antecedent. There is therefore no situation where the model can correctly predict "no antecedent," so true negatives do not occur.

   What causes FN
   - the correct antecedent is scored lower than a competing NP (e.g. the true mention is earlier in the text while a closer NP wins)
   - extraction failures: the correct NP wasn't extracted at all before the pronoun
   - name-form mismatch: the dataset uses a long form but extraction returns a shortened variant ("Antonio di Filippo…" vs "Antonio")

   Examples
   - test-23: Pronoun 'her' — true 'Joyce', predicted 'Lisa' (closer competing NP)
   - test-25: Pronoun 'his' — true 'Oliver Turvey', predicted 'Giedo' (competing nearby person)
   - test-36: Pronoun 'her' — true 'Anais', predicted 'Deluxe Edition' (correct NP present but noisy extraction)

   What causes FP
   - a wrong candidate looks better on surface cues (distance, gender/number) — e.g. "patient" chosen over "Dr. Smith"
   - noisy extraction picks non-entity tokens as NPs (section titles, abbreviations like "Mass")
   - several near-identical candidates (same gender/number and similar distance) make the model choose the wrong one

   Examples
   - test-8: Pronoun 'her' — true 'Jeni', predicted 'The monster' (non-human mention treated as candidate)
   - test-9: Pronoun 'he' — true 'Malave', predicted 'Mass' (abbreviation/section picked as NP)
   - test-6: Pronoun 'her' — true 'Jewel Staite', predicted 'Morena Baccarin' (multiple nearby person names)

   How to improve
   - add semantic/contextual signals (NER, embeddings, animacy) rather than just surface features
   - switch to a direct ranking or pairwise model instead of independent NB scores
   - improve NP extraction (dependency-based NPs, NER) and use contextual embeddings (BERT) for richer features
