### word length vs freq
| Word Length | Frequency |
|------------:|----------:|
| 1           | 22471     |
| 2           | 140631    |
| 3           | 236862    |
| 4           | 188422    |
| 5           | 102704    |
| 6           | 57927     |
| 7           | 42937     |
| 8           | 26519     |
| 9           | 17927     |
|10           | 8108      |
|11           | 4014      |
|12           | 1846      |
|13           | 929       |
|14           | 367       |
|15           | 95        |
|16           | 17        |
|17           | 4         |
|18           | 2         |

### top 50 words that appeared the most-
- ('the', 68049)
- ('and', 54114)
- ('of', 36197)
- ('to', 14974)
- ('that', 13610)
- ('in', 13477)
- ('he', 11519)
- ('shall', 9885)
- ('i', 9512)
- ('for', 9464)
- ('a', 9408)
- ('his', 9132)
- ('unto', 8997)
- ('lord', 7968)
- ('they', 7764)
- ('is', 7475)
- ('be', 7235)
- ('him', 7032)
- ('not', 6941)
- ('them', 6574)
- ('it', 6555)
- ('with', 6495)
- ('all', 5992)
- ('thou', 5572)
- ('was', 5059)
- ('thy', 4663)
- ('my', 4558)
- ('god', 4475)
- ('which', 4457)
- ('said', 4441)
- ('but', 4337)
- ('me', 4259)
- ('have', 4153)
- ('their', 4080)
- ('ye', 4037)
- ('will', 4025)
- ('as', 3958)
- ('thee', 3904)
- ('from', 3826)
- ('are', 3173)
- ('you', 3052)
- ('when', 3049)
- ('this', 2996)
- ('out', 2987)
- ('were', 2944)
- ('man', 2915)
- ('by', 2854)
- ('upon', 2777)
- ('up', 2664)
- ('israel', 2575)

### pearson coefficient: -0.7412284934081025

### short note on "are word lengths optimized for effective communication"
- Yes, the given research paper suggests word length isnt random. It shows that longer words are usually less predictable and carry more information, while shorter ones tend to appear in highly predictable contexts.

- Its not just about how often a word is used. 
Earlier ideas (zipf's law) argued that frequent words become short over time. instead, a word’s information content is a better explanation for its length.

- Using n-gram models to measure how surprising a word is in context, the authors found that longer words are typically used when more clarity is needed and guessing is harder, suggesting that there exists a communicative purpose behind their length.

- Across 11 languages, the same pattern appears- shorter words do just enough work to be understood in context, while longer words step in to avoid confusion when more information must be conveyed. This “effort vs clarity” balance points to optimized design.

- Humans naturally shorten words that are easy to predict and keep words longer when they need to deliver precise information. 

### shortest words in the dataset
| Word | Frequency |
|-----:|----------:|
| i     | 9512     |
| a     | 9408     |
| s     | 2189     |
| o     | 1105     |
| t     | 110      |
| e     | 47       |
| m     | 25       |
| f     | 22       |
| u     | 14       |
| d     | 12       |
| c     | 8        |
| h     | 8        |
| b     | 4        |
| l     | 3        |
| k     | 2        |
| n     | 2        |

given above are all words with the min possible length found in my dataset. 
"I" and "a" appear the most out of them by a huge margin. this is because I is a pronoun and a is an indefinite article which are both function words which are used very commonly.
the appearance of "s", "t" and "o" is due to my tokenisation method, which separates words at punctuations, like apostrophes. "o" appears in situations like "o'clock" and O being used as an alternative to "oh" (this was common in old english)