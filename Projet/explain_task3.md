# Create explanation .md file
explanation = """\
# Enhancements for Improved Classification (SciTweets Dataset)

This markdown file explains the key techniques used to improve the accuracy and F1 score of the multi-label classification model for the `scitweets_export.tsv` dataset.

---

## 1. Text Preprocessing

**Purpose**: Normalize the input to reduce noise and improve semantic similarity extraction.

- Lowercasing all characters.
- Replacing URLs with "URL", mentions with "MENTION", and removing special characters.
- Hashtag words are kept but the hash symbol is removed.

---

## 2. Sentence Embeddings

**Tool**: `paraphrase-multilingual-MiniLM-L12-v2` (from SentenceTransformers)

**Why**:
- Captures semantic meaning of the tweets.
- Handles multilingual content (important since tweets may contain both French and English).

---

## 3. Additional Metadata Features

These handcrafted features enrich the representation of each tweet:

- `has_url`: Binary indicator if tweet contains a URL.
- `has_mention`: Binary indicator for mentions.
- `has_hashtag`: Binary indicator for hashtags.
- `text_len`: Character length of the tweet.

These simple features often correlate with the likelihood of a tweet being a claim, ref, or context.

---

## 4. Feature Combination

- Sentence embeddings (dense semantic features) are concatenated with the handcrafted binary features and length.
- This provides the model with both high-level context and low-level signals.

---

## 5. Standardization

**Tool**: `StandardScaler`

- Standardizing all features ensures the model isn't biased toward longer embeddings or larger-scale inputs.

---

## 6. Multi-Label Classification Strategy

**Technique**: `ClassifierChain` with `LogisticRegression`

- Allows label dependencies to be learned (e.g., some tweets may be both `context` and `reference`).
- `LogisticRegression` offers interpretability and performs well in high-dimensional settings.

---

## 7. Evaluation

**Metric**: `classification_report` (includes F1-score, precision, recall per label)

---

This combination of semantic embeddings, metadata enrichment, preprocessing, and multi-label modeling offers a strong baseline for further improvements.
"""

# Write to a .md file
explanation_path = "/mnt/data/sci_tweet_classifier_explanation.md"
with open(explanation_path, "w") as f:
    f.write(explanation)

explanation_path
