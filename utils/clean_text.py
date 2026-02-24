import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_slug(url):
    path = urlparse(url).path
    slug = path.strip("/")
    slug = re.sub(r"\.(jpg|jpeg|png|gif)$", "", slug)
    slug = slug.replace("-", " ").replace("_", " ")
    return slug


def clean_text(text):

    text = text.lower()

    # Extract URLs first
    urls = re.findall(r"https?://\S+", text)

    slug_text = ""
    for url in urls:
        slug_text += " " + extract_slug(url)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", text)

    # Combine original text + extracted slug text
    text = text + " " + slug_text

    # Remove special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def get_wordnet_pos(tag):
    #Maps it to a WordNet-compatible tag
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
    }
    return tag_dict.get(tag[0], wordnet.NOUN) # returns the word type (Noun if we have not found)

def preprocess_text(text):
    text = clean_text(text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_set]

    tagged_tokens = nltk.pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]

    return " ".join(lemmatized)