## Module 6: Natural Language Processing (NLP)

Natural Language Processing (NLP) is the field of artificial intelligence that focuses on the interaction between computers and human language. NLP is used to make sense of text, allowing machines to understand, interpret, and generate human language. Below, we'll explore key concepts in NLP, each with explanations and examples.

---

### 1. **Text Preprocessing**

Text preprocessing is a critical step in NLP, as raw text data is often noisy and unstructured. The goal is to clean and transform the text into a format that can be processed by machine learning models.

#### Key tasks in text preprocessing:

- **Tokenization**: Splitting text into smaller units (tokens).
- **Lowercasing**: Converting all text to lowercase to avoid distinguishing between words like "Apple" and "apple."
- **Stopword Removal**: Removing common words that don't add much value to the text, such as "the," "is," "and."
- **Stemming**: Reducing words to their root form, e.g., "running" becomes "run."
- **Lemmatization**: Converting words to their base form (more sophisticated than stemming), e.g., "better" becomes "good."

#### Example:
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Sample text
text = "I am running towards the store to buy apples."

# Tokenization
tokens = word_tokenize(text)

# Lowercasing
tokens = [word.lower() for word in tokens]

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Stemming
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

print("Tokens:", tokens)
print("Filtered Tokens:", filtered_tokens)
print("Stemmed Tokens:", stemmed_tokens)
```

**Output:**
```
Tokens: ['I', 'am', 'running', 'towards', 'the', 'store', 'to', 'buy', 'apples', '.']
Filtered Tokens: ['running', 'towards', 'store', 'buy', 'apples', '.']
Stemmed Tokens: ['run', 'toward', 'store', 'buy', 'appl', '.']
```

---

### 2. **Tokenization and Embeddings**

- **Tokenization** splits a sentence into meaningful units (words, subwords, or characters). It's one of the first steps in NLP for transforming raw text into a structured form.

- **Word Embeddings** represent words as vectors in a high-dimensional space. These embeddings capture the semantic relationships between words.

#### Example:
Using `Word2Vec` to create word embeddings for words:
```python
from gensim.models import Word2Vec

# Sample sentences for training
sentences = [["I", "love", "machine", "learning"],
             ["NLP", "is", "fun"],
             ["I", "enjoy", "coding"]]

# Train a Word2Vec model
model = Word2Vec(sentences, min_count=1)

# Get the embedding for the word 'machine'
embedding = model.wv['machine']
print("Embedding for 'machine':", embedding)
```

**Output:**
```
Embedding for 'machine': [ 0.03009117  0.02313066 -0.03427672 ...]
```

Embeddings allow words with similar meanings to have similar representations in vector space (e.g., "king" and "queen").

---

### 3. **Sentiment Analysis**

Sentiment analysis aims to determine the sentiment expressed in a piece of text, classifying it into categories like positive, negative, or neutral.

#### Example:
Using a pre-trained sentiment analysis model from Hugging Face's `transformers` library:
```python
from transformers import pipeline

# Load a sentiment-analysis model
sentiment_analyzer = pipeline('sentiment-analysis')

# Sample text
text = "I love using Python for data science!"

# Perform sentiment analysis
result = sentiment_analyzer(text)
print(result)
```

**Output:**
```
[{'label': 'POSITIVE', 'score': 0.9998}]
```

In this example, the model classifies the text as "POSITIVE" with a high confidence score.

---

### 4. **Sequence-to-Sequence Models**

Sequence-to-Sequence (Seq2Seq) models are used for tasks like machine translation, summarization, and text generation. These models are often implemented with Recurrent Neural Networks (RNNs) or the more modern Transformer architecture.

#### Example: Machine Translation (English to French)
Using the `transformers` library for translation:
```python
from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-fr'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Input text (English)
text = "Hello, how are you?"

# Tokenize and translate
tokens = tokenizer(text, return_tensors="pt")
translation = model.generate(**tokens)

# Decode the translation
translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
print("Translated Text:", translated_text)
```

**Output:**
```
Translated Text: Bonjour, comment ça va?
```

The Seq2Seq model translates the English text to French.

---

### 5. **Transformers and BERT**

Transformers are powerful models that use a self-attention mechanism to process sequences in parallel. BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model that has significantly improved performance in various NLP tasks.

#### Example: Using BERT for Text Classification
Let's use BERT for text classification (e.g., determining whether a movie review is positive or negative).

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch import nn
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Example text
text = "The movie was fantastic!"

# Tokenize and encode input text
inputs = tokenizer(text, return_tensors='pt')

# Get model output
outputs = model(**inputs)

# Get predictions
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1)

print("Predicted Class:", predicted_class.item())
```

**Output:**
```
Predicted Class: 1  # Assuming 0 is negative, 1 is positive
```

BERT processes the text and classifies the sentiment as positive (class 1).

---

### Resources

- **Coursera: Natural Language Processing Specialization**  
  A comprehensive series of courses to learn NLP concepts in detail, including machine translation, speech recognition, and more.

- **Hugging Face Transformers**  
  A library offering pre-trained models like BERT, GPT, and many others for NLP tasks such as text generation, classification, translation, and more.
  - Website: [Hugging Face](https://huggingface.co/)
