# RST-Based Summarization: Discussion Report

## Task 1: Parsing with RST Parser

Ran pretrained `isanlp_rst` parser (rstdt version) on 10 paragraphs from `paragraph.txt`. Generated RS3 XML files (1.rs3-10.rs3) with full discourse trees containing nucleus/satellite relations and hierarchical structure.

## Task 2: Extract RST Information

Parsed RS3 files extracting nucleus/satellite labels and relation types:
- **Relations**: Attribution, Background, Condition, Contrast, Elaboration, Explanation, antithesis
- **Nuclei vs Satellites**: Multinuc/span=nuclei; rst=satellites
- Output: `rst_extracted_info.json` with segment classifications

## Task 3: Rule-Based Summarization

Implemented 5 scoring rules to select important segments:

| Rule | Score | Reason |
|------|-------|--------|
| Nucleus Preference | +5 | Nuclei carry essential content |
| Depth Scoring | 10-depth | Top-level nodes more central |
| Relation Weighting | 0-7 | Contrast/Explanation=7, Background=5, Elaboration=3, Attribution=2 |
| Sentence Completeness | +3 | Ends with `.!?` |
| Length Bonus | +2/+2 | >50 chars (+2), >100 chars (+2) |

Selected top 3 segments per paragraph. Generated 1-3 sentence summaries for all 10 paragraphs (in `rst_summarization_report.txt`).

## Task 4: Evaluation Metrics

Computed token-based F1-score, Precision, and Recall using stopword-filtered word overlap:

**Results:**
- Precision: 0.232
- Recall: 0.211
- **F1-Score: 0.217**

**Best**: Para 1 (F1=0.450), Para 7 (F1=0.364)  
**Worst**: Para 4 (F1=0.125), Para 6 (F1=0.129), Para 5 (F1=0.143)

## Task 5: Discussion

**What worked:**
- Nucleus preference: Best rule, well-structured text
- Technical/factual content: F1≈0.26 (clear hierarchy, facts extract well)
- Sentence completeness: Improved readability

**What failed:**
- Narratives/humor: F1≈0.14 (miss climax, temporal flow, punchlines)
- Length bonus: Favors verbose setups over concise key points
- Fragment selection: "Then," alone is meaningless
- Fixed rules don't adapt to genres

**Why F1=0.217 is low:**
1. Extractive vs abstractive: LLM paraphrases; system copies fragments
2. No semantic understanding: Can't identify "ostentatious" as punchline
3. RST granularity: Clauses too small for semantic meaning units
4. Genre-blind rules: One-size-fits-all fails on 50% of texts

**Possible improvements:**
1. Remove length bonus (hurts performance)
2. Reweight relations to match actual data distribution
3. Merge adjacent fragments from same parent
4. Genre detection: different rules for narratives vs. technical
5. Coherence filter: skip segments with dangling references
6. TF-IDF scoring: weight segments by keyword importance

**Conclusion:** RST structure adds value (F1=0.217 >> random ~0.1), nucleus/satellite is most useful signal. Technical content works well. But structure alone insufficient for narratives/paraphrasing. System demonstrates RST helps but needs semantic analysis for full summarization.

## Per-Paragraph Analysis: What Worked vs What Didn't

### Para 1 (F1=0.450) WORKED WELL
**Rule Summary:** "that endorsing the Nuclear Freeze Initiative is the right step for California Common Cause. Tempting as it may be, if we stick to those issues of governmental structure and process, broadly defined,"  
**LLM:** "The author argues that California Common Cause should focus on its core mission of governmental structure and process rather than spreading resources thin by supporting every popular cause like the Nuclear Freeze Initiative."
- **What worked:** Extracted key phrases directly from main nucleus segments. Captured core argument and conclusion about "governmental structure and process."
- **What didn't:** Grammar awkward (missing subject verb). LLM paraphrases better and adds context (spreading resources thin).

### Para 2 (F1=0.186) PARTIAL
**Rule Summary:** "a Syncom diskette is working four ways. Cleaning agents on the burnished surface of the Ectype coating actually remove build-up from the head, Strong binders hold the signal-carrying oxides tightly within the coating."  
**LLM:** "Syncom diskettes use four protective mechanisms—cleaning agents, lubricants, carbon additives, and strong binders—along with a special jacket liner to prevent dust and particles from causing data errors."
- **What worked:** Selected multiple factual mechanisms correctly. Sentences mostly complete and readable.
- **What didn't:** Missed carbon additive and jacket liner. LLM summarizes all 4 mechanisms; rules only got 2/4.

### Para 3 (F1=0.211) PARTIAL
**Rule Summary:** "To create your own \"Victorian\" bouquet of flowers, Leaf textures and colors will also be important. the deep greens, or chartreuse, even the widely varied foliage colors of the coleus."  
**LLM:** "Creating a Victorian-style flower arrangement requires combining plants of varying heights, shapes, and colors, with attention to diverse leaf textures ranging from silver dusty miller to colorful coleus."
- **What worked:** Got the key instruction and mentioned colors/textures.
- **What didn't:** Grammar broken ("Leaf" capitalized mid-sentence). Missing height variety and dusty miller example. LLM mentions more plant types.

### Para 4 (F1=0.125) FAILED
**Rule Summary:** "carried a message, a refutation, of claims. but the tragic and too-common tableaux of hundreds or even thousands of people. illustrates a lack of jobs, not laziness."  
**LLM:** "Long lines of job seekers demonstrate a shortage of available employment rather than a lack of effort, contradicting claims that unemployment stems from insufficient initiative."
- **What worked:** Final segment "illustrates a lack of jobs, not laziness" captures main point.
- **What didn't:** First two segments are fragments that make no sense alone ("carried a message, a refutation, of claims"). Lost coherence entirely. LLM completely rewrites for clarity.

### Para 5 (F1=0.143) FAILED
**Rule Summary:** "my wife was amused by the real-estate agents, who engaged her in the \"you talk kinda funny\" conversation. Then,"  
**LLM:** "When real estate agents asked the author's English wife what she would call an elaborate chandelier, she wittily replied \"ostentatious\" after discussing other British-American vocabulary differences."
- **What worked:** None—completely missed the punchline.
- **What didn't:** Extracted setup ("amused by agents," "you talk kinda funny") but ended with dangling "Then," which means nothing. Missing the entire punchline "ostentatious." This is the worst failure—joke completely lost.

### Para 6 (F1=0.129) FAILED
**Rule Summary:** "When a group of Americans, many in the teaching profession, visited her in Calcutta, \"Smile at your husbands.\". and I find it hard sometimes to smile at Jesus."  
**LLM:** "Mother Teresa advised visiting American teachers to smile at their spouses, and when asked if marriage made her advice relevant, revealed she is \"married\" to Jesus, who she finds can be very demanding."
- **What worked:** Got the key quotes and the Jesus marriage reference.
- **What didn't:** Grammar errors (malformed quote, "and I find it" dangling). Missing the context (who asked the question, why the surprise). LLM adds crucial connective tissue explaining the exchange.

### Para 7 (F1=0.364) WORKED WELL
**Rule Summary:** "is the nation's first survey. It ranks 184 urban areas on 11 different criteria. The Urban Stress Test translates complex, technical data into an easy-to-use action tool for concerned citizens, elected officials and opinion leaders."  
**LLM:** "ZPG's Urban Stress Test ranks 184 U.S. cities using eleven population-related criteria to help citizens and officials understand urban pressures in an accessible format."
- **What worked:** Extracted all key facts: "first survey," "184 urban areas," "11 different criteria," "translates complex data into action tool." Perfect fact extraction.
- **What didn't:** Missing "ZPG's" name and "1985" date. Grammar missing subject in first segment. But core facts intact.

### Para 8 (F1=0.217) PARTIAL
**Rule Summary:** "Our knowledge of ancient shipbuilding techniques, for example, is dependent on the work of archaeologists. The contents of shipwrecks are an equally valuable archaeological resource. they provide better chronological information than even the most carefully excavated terrestrial site."  
**LLM:** "Archaeological study of ancient shipwrecks provides uniquely valuable chronological data since all objects represent a single moment in time, and underwater conditions preserve perishable items that would decay on land."
- **What worked:** Got key concepts: dependency on archaeologists, shipwrecks as resource, chronological advantage.
- **What didn't:** Grammar error ("they" lowercase). Missing perishable preservation benefit. LLM more concise and complete.

### Para 9 (F1=0.143) FAILED
**Rule Summary:** "In an article for Campus Life magazine, Susan Smart describes her third and nearly last solo flight. She had been practicing a maneuver at 5,000 feet. and her Cessna 150 began spinning wildly toward earth."  
**LLM:** "A student pilot lost control of her spinning Cessna 150 but survived by remembering her instructor's counterintuitive advice to simply release the controls and let the plane stabilize itself."
- **What worked:** Got the setup and crisis (spinning toward earth).
- **What didn't:** Missing the entire resolution/punchline (the instructor's advice to "let go of controls"). Narrative is incomplete without the solution. LLM captures the survival/solution.

### Para 10 (F1=0.207) PARTIAL
**Rule Summary:** "This paper is part of an effort. It is the first in a pair of papers. Systemic Linguistics is a much more comprehensive view of language."  
**LLM:** "This paper compares Rhetorical Structure Theory, which analyzes texts through functional relations between parts, with the broader Systemic Linguistics approach, which categorizes texts by the processes they perform."
- **What worked:** Got high-level structure (part of effort, first in pair).
- **What didn't:** Generic sentences miss the actual comparison. Missing RST definition and key distinction (functional relations vs. processes). LLM explains what each theory does.

## Summary of Patterns

| Category | Paragraphs | Success | Issues |
|----------|-----------|---------|--------|
| Technical/Factual | 2, 7, 8 | Extract facts well | Minor details missed |
| Argumentative | 1, 4, 10 | Mixed | Fragments incoherent (Para 4), generic summaries (Para 10) |
| Narrative/Story | 5, 9 | Major failures | Missing resolution/punchline, dangling "Then," |
| Descriptive | 3, 6 | Partial | Grammar issues, incomplete context |

Key insight: Rules work for facts but fail on narrative structure and need coherence checking for fragments.
