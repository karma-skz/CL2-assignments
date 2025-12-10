import re

def remove_extra_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+(?=[.,,"\'!?;:])', '', text)
    return text.strip()

def to_lowercase(text):
    return text.lower()

def remove_special_characters(text):
    # Keep basic punctuation that's important for sentiment: . , ! ? ' " : ;
    return re.sub(r'[^a-zA-Z0-9.,!?\'";:\s<>]+', '', text)

def handle_common_abbreviations(text):
    contractions_patterns = [
        (r"can't", "can not"),
        (r"shan't", "shall not"), 
        (r"won't", "will not"),
        (r"n't", " not"),
        (r"'m", " am"),
        (r"'re", " are"),
        (r"'d", " would"),
        (r"'ll", " will"),
        (r"'ve", " have"),
        (r"he's", "he is"),
        (r"she's", "she is"),
        (r"it's", "it is"),
        (r"that's", "that is"),
        (r"what's", "what is"),
        (r"there's", "there is"),
        (r"here's", "here is"),
        (r"where's", "where is"),
        (r"when's", "when is"),
        (r"why's", "why is"),
        (r"who's", "who is"),
        (r"how's", "how is"),
        (r"let's", "let us"),
    ]
    for pattern, replacement in contractions_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def remove_html_tags(text):
    text = re.sub(r'<[^>]+>', '', text)
    return text

def replace_tokens_with_placeholders(text):
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    text = re.sub(email_pattern, '<MAIL>', text)
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = re.sub(url_pattern, '<URL>', text)
    hashtag_pattern = re.compile(r'#[a-zA-Z0-9_+&!]+')
    text = re.sub(hashtag_pattern, '<HASHTAG>', text)
    mention_pattern = re.compile(r'@[a-zA-Z0-9_+&!]+')
    text = re.sub(mention_pattern, '<MENTION>', text)
    return text

def standardize_quotation_marks(text):
    text = re.sub(r"(?<!\w)`+|`+(?!\w)", '"', text)
    text = re.sub(r"(?<!\w)'+|'+(?!\w)", '"', text)
    text = re.sub(r'''(?<!\w)"+|"+(?!\w)''', '"', text)
    return text

def normalise_ellipses(text):
    ellipses = re.compile(r'\.{2,}')
    text = re.sub(ellipses, "...", text)
    return text

def remove_numbers(text):
    return re.sub(r'\b\d+\b', '', text)

def normalize_repeated_punctuation(text):
    text = re.sub(r'!{2,}', '!', text)  # Multiple ! to single !
    text = re.sub(r'\?{2,}', '?', text)  # Multiple ? to single ?
    text = re.sub(r'\.{2,}', '...', text)  # Multiple . to ...
    return text

def clean(text):
    text = remove_html_tags(text)
    text = to_lowercase(text)
    text = normalize_repeated_punctuation(text)
    text = handle_common_abbreviations(text)
    text = standardize_quotation_marks(text)
    text = normalise_ellipses(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)    
    return text

def kam_cleaning(text):
    text = remove_html_tags(text)
    text = to_lowercase(text)
    text = normalize_repeated_punctuation(text)
    # text = handle_common_abbreviations(text)
    text = standardize_quotation_marks(text)
    text = normalise_ellipses(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)    
    return text
