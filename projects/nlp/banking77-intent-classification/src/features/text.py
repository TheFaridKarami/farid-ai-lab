from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextVectorizer:
    
    def __init__(self, config: dict, method: str) ->None:
        self.method = method
        self.config = config
        self.vectorizer = self._create_vectorizer()
        
        
    def _create_vectorizer(self) -> CountVectorizer | TfidfVectorizer:
        
        if self.method == "bow":
            
            config = self.config["bow"].copy()
            config["ngram_range"] = tuple(config["ngram_range"])
            
            return CountVectorizer(**config)
        
        
        elif self.method =="tfidf":
            
            config = self.config["tfidf"].copy()
            config["ngram_range"] = tuple(config["ngram_range"])
            
            return TfidfVectorizer(**config)
        
        else:
            
            raise ValueError(
                f"Unsupported method: {self.method!r}. "
                "Accepted methods: 'bow' and 'tfidf'."
            )
            
            
    def fit_transform(self, texts:list[str]):
        
        return self.vectorizer.fit_transform(texts)
        
        
    def transform(self, texts:list[str]):
        
        return self.vectorizer.transform(texts)
        
    
        

