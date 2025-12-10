# IMP this should be the first thing you read!
## File structure-
### `Language_datasets.zip` -> the data provided to us
extract it inside the folder and just run both py scripts.

`The metrics recorded`-
-   ├── result_pol_ces.md
-   ├── result_por_spa.md

(generate the metrics by running scripts)

# Notes on Character ngram Classifier Performance

## N-grams with Zero Count in the Test Set
The main reason is out of vocabulary words, which give a zero probability, and mess up the scores for the sentence. I have not implemented any smoothing techniques in this task, like laplace smoothing but that is how to solve this problem. intuitively, it takes adds 1 count to every word in the dataset including the OOV gram to make it non zero.

## Spanish vs Portuguese  

### Misclassifications  
- The system often failed with short sentences. For example:  
  - "¡Completamente seguro!" (Spanish) was read as Portuguese once the "¡" was removed, because nothing special like ñ remained.  
  - "Tom animó a Mary." had common names and common endings that made it drift to Portuguese even though the accent in "animó" pointed to Spanish.  
  - "¿Querés compartirlo?" has the word "querés" which is Spanish-only, but since it was short and punctuation was stripped, the machine leaned Portuguese.  
- On the other side:  
  - "Falas eslovaco?" (Portuguese) was tagged as Spanish because the letter groups also fit Spanish.  
  - "És enorme." was too short and the accent alone was not enough to anchor Portuguese.  

So the errors mostly came from short, common shapes without strong unique letters.  

### Successful cases  
- When longer word pieces appeared, the model used distinctive clusters:  
  - Portuguese: **nh, lh, ão, ção, você**  
  - Spanish: **ñ, ll, -ción, -aron/-ieron, clitic pairs like "me lo"**  
- These longer n-grams (3–4 characters) gave the system a stable fingerprint for each language, which explains the big jump in accuracy.  

---

## Polish vs Czech  

### Misclassifications  
- Short texts were the main issue:  
  - "Pes, kočka a medvěd…" (Czech) was misread as Polish even though letters like č, ě were there, because overlapping grams like "na" and "je" pulled it.  
  - "Můj švára je policajt." was a casual Czech phrase, but many grams are shared so the anchor signals were weak.  
  - On the Polish side: "Czyje to?" was short with no strong Polish-only letters, so it slipped into Czech.  
- In short, too few special letters and too many common ones led to confusion.  

### Successful cases  
- With longer strings, special letters stood out:  
  - Polish: **ą, ę, ł, ó, ś, ć, ż** in clusters like *"łą"*, *" sł"*, *"ówi"*  
  - Czech: **ř, ě, ů, ť, ď, ň** in endings like *"-ovat"*, *"-něji"*  
- These distinctive combos at 3–4 letters gave the system very high accuracy close to 0.95.  

---

## Summarising  

- Errors came from short, high-overlap texts with missing or weak unique markers.  
- Successes came from longer pieces where special letters or endings (**ñ vs ão**, **ł vs ř**) showed up.  
- In both language pairs the machine got confused in the same way and became accurate once the text was long enough to reveal the distinct patterns.  


---
## Assumptions and preprocessing

- Encoding and Unicode
  - All files are read/written in UTF-8.
  - Text is normalized to Unicode NFC and lowercased. (Why use NFC- https://www.macchiato.com/unicode-intl-sw/nfc-faq)
  - I removed NFC seeing that it didnt change the metrics at all, maybe because a lot of the diacritics are specific to a particular language only so it doesnt change anything much.
- Normalization and filtering
  - Only alphabetic characters are kept. digits and punctuation are dropped. Diacritics are preserved (eg, ñ, ã, ł, č remain because they are alphabetic).
  - Whitespace is collapsed to single spaces and leading/trailing spaces are trimmed.
  - Effect: Spanish inverted punctuation (¿ ¡), quotes, hyphens, etc., are removed and thus do not contribute to ngrams. I figured that keeping punctuation had a bad effect on the model's accuracy that outweighed the benefit of keeping spanish only punctuation.

- Tokenization and ngram generation
  - Space characters are preserved and therefore can appear inside character ngrams.
  - No start/end boundary markers are added as we have been asked to calculate character ngrams.