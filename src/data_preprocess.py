import re
import nltk
from urllib.parse import urlparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def extract_slug(self, url):
        path = urlparse(url).path
        slug = path.strip("/")
        slug = re.sub(r"\.(jpg|jpeg|png|gif)$", "", slug)
        slug = slug.replace("-", " ").replace("_", " ")
        return slug
    
    # Assign POS 
    def get_wordnet_pos(self, tag):
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag[0], wordnet.NOUN)

    def clean_text(self, text):

        text = text.lower()                                         # Apply lowercase

        urls = re.findall(r"https?://\S+", text)                    # Extract URLs
        slug_text = " "
        for url in urls:
            slug_text += " " + self.extract_slug(url)

        text = re.sub(r"https?://\S+", " ", text)                    # Remove URLs
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)      # Remove emails
        text = text + " " + slug_text                                # Combine original + slug
    
        text = re.sub(r"[^a-z\s]", " ", text)                       # Keep only letters
        text = re.sub(r"\s+", " ", text).strip()                    # Strip unnecessary spaces
        
        return text

    def preprocess(self, text):

        # if not isinstance(text, str):
        #     return ""

        # text = self.clean_text(text)

        # if not text.strip():
        #     return ""
        
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]

        tagged_tokens = nltk.pos_tag(tokens)

        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
            for word, tag in tagged_tokens
        ]

        return " ".join(lemmatized)