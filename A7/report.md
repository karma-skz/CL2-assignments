# Assignment-6: Co-occurrence Matrix, SVD and Similarity Analysis

## Word Pair Selection (Section 2.3)

The following 5 word pairs were selected to evaluate the embeddings. These pairs represent different semantic relationships:

1.  **'space' and 'planet'**: Represents a broad, space-related context (Related Concepts).
2.  **'astronauts' and 'robot'**: Technology and space exploration-related entities (Related but Distinct).
3.  **'more' and 'less'**: Represents antonyms.
4.  **'scientists' and 'researchers'**: Represents near-synonyms.
5.  **'rocket' and 'telescope'**: Both related to space exploration but with different roles (Domain Related).

## Similarity Scores (Section 2.4)

The calculated cosine similarity scores for the chosen word pairs across four different window sizes are reported below:

| Word Pair | Window Size 1 | Window Size 2 | Window Size 3 | Window Size 4 |
| :--- | :--- | :--- | :--- | :--- |
| **'space' and 'planet'** | 0.2557 | 0.3155 | 0.2598 | 0.1425 |
| **'astronauts' and 'robot'** | -0.0514 | 0.1032 | -0.0991 | 0.0938 |
| **'more' and 'less'** | 0.9250 | 0.9326 | 0.9044 | 0.8820 |
| **'scientists' and 'researchers'** | 0.9306 | 0.9187 | 0.8714 | 0.8886 |
| **'rocket' and 'telescope'** | 0.2680 | 0.3088 | 0.4464 | 0.1877 |

## Comparison of Context Windows (Section 2.4)

The results show how similarity scores vary with different context window sizes:

*   **Smaller Context Windows (1-2):** These windows capture local, direct relationships. For *'space'* and *'planet'*, the relationship peaks at window size 2 (0.3155), suggesting they often appear in close proximity but not necessarily immediately adjacent.
*   **Larger Context Windows (3-4):** Larger windows allow for capturing global relationships, but can also introduce noise. For *'rocket'* and *'telescope'*, the similarity peaks at window size 3 (0.4464) before dropping significantly at window size 4, indicating the optimal context for capturing their relationship is around 3 words.

## Analysis of Findings (Section 2.4)

### Analysis of Similarity Scores
Higher cosine similarity scores (closer to 1) indicate a stronger semantic relationship between the word vectors.
*   **Strong Relationships:** Pairs like *'more' vs 'less'* and *'scientists' vs 'researchers'* consistently show very high similarity scores (> 0.87) across all window sizes. This indicates that these antonyms and synonyms share very similar contexts regardless of the window size.
*   **Fluctuating/Weak Relationships:** Pairs like *'astronauts' vs 'robot'* show significant fluctuation, alternating between positive and negative scores. This suggests a lack of consistent co-occurrence patterns in this specific dataset, or that their relationship is highly dependent on specific sentence structures.

### Impact of Context Window Size
Changing the context window size impacts the embeddings and resulting similarity scores:
1.  **'space' and 'planet'**: The score increases from Window 1 to Window 2, then decreases. This suggests a "sweet spot" at Window 2 where the semantic relationship is best captured.
2.  **'rocket' and 'telescope'**: The similarity increases up to Window 3, then drops. This implies that these terms are often separated by a few words, but a window of 4 becomes too broad, introducing unrelated context.
3.  **'scientists' and 'researchers'**: There is a slight general decline as the window size increases, but the scores remain robustly high, confirming their strong synonymy.
4.  **'more' and 'less'**: Similar to the synonyms, these antonyms maintain high similarity, peaking slightly at Window 2.

**Significant Differences:**
*   **'astronauts' and 'robot'**: This pair shows the most erratic behavior, flipping signs between window sizes. This indicates that the model struggles to find a stable relationship between these terms with the given data and parameters.
*   **'rocket' and 'telescope'**: The jump from ~0.27 (Window 1) to ~0.45 (Window 3) highlights that a slightly larger context helps capture the relationship between these domain-specific tools.
