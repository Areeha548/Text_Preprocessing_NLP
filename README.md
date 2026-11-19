# Text Preprocessing for NLP (English + Urdu)

A beginner-friendly Jupyter notebook that shows how to turn messy human text into clean tokens and then into numbers a machine learning model can use. Every step is explained first and then run in code. English and Urdu examples run side by side.

This repo also contains my solutions to the **Section 12 exercises** at the end of the notebook.

**Notebook:** `Areeha_Shafique_Text_Preprocessing_NLP_Solved.ipynb`

---

## Why preprocess text?

Models understand numbers, not words. If we feed raw text, `Good`, `good` and `GOOD` look like three different words. Preprocessing cleans the text so the model sees one consistent word.

The pipeline built in this notebook:

```
clean → lowercase / normalize → tokenize → remove stop words → stem / lemmatize → vectors
```

Urdu is a good stress test because it is written right to left, has no upper/lower case, uses optional diacritics (aeraab), and changes word endings a lot.

---

## What is inside the notebook

| # | Topic | What it does | Small example |
|---|-------|--------------|---------------|
| 0 | Setup | Imports libraries and checks which optional ones are installed | `numpy`, `sklearn`, `nltk` |
| 2 | Cleaning and normalization | Removes HTML, URLs, emails. For Urdu, unifies letter variants and removes diacritics | `Visit https://x.com! Email a@b.com <b>NOW</b>` becomes `Visit Email NOW` |
| 3 | Lowercasing | Makes all English letters small | `The Cat SAT` becomes `the cat sat` |
| 4 | Tokenization | Splits text into words (naive split, regex, NLTK) | `"I don't love NLP."` split three ways |
| 5 | Stop-word removal | Drops very common words like `the`, `of`, `کا`, `ہے` | `['i','love','the','game','of','cricket']` becomes `['love','game','cricket']` |
| 6 | Stemming | Cuts word endings with rules. Fast but crude | `studies` becomes `studi` |
| 7 | Lemmatization | Maps a word to its real dictionary form | `studies` becomes `study`, `better` becomes `good` |
| 8 | Full pipeline | Chains all steps for English and Urdu | See notebook |
| 9 | One-hot encoding | One vector per word with a single `1` | `cat` = `[1, 0, 0, 0]` |
| 10 | Bag of Words and TF-IDF | Counts words per document, then weights rare words higher | See exercise 4 |
| 11 | What comes next | Embeddings (Word2Vec, GloVe, fastText, BERT) | Discussion only |

---

## Exercise solutions

### 1. Stemming vs lemmatization (English)
I made `preprocess_en_stem` (Porter stemmer) and compared it with the lemmatizer version on this sentence:
`"The students are studying and the cats were running happily. Studies show better organization."`

| Method | Output |
|--------|--------|
| Stemming | `student, studi, cat, were, run, happili, studi, show, better, organ` |
| Lemmatization | `students, study, cat, be, run, happily, study, show, better, organization` |

**Takeaway:** stemming can create words that do not exist (`studi`, `happili`, `organ`) and left `were` unchanged. Lemmatization always gives real words and turned `were` into `be`. Stemming is faster, lemmatization is more accurate.

### 2. Adding three Urdu stop words
I added **بھی**, **کہ** and **تو**. The stop-word list grew from 17 to 20 words.
Test sentence: `وہ بھی جانتا ہے کہ لڑکے میدان میں کھیل رہے تھے تو`
Result: `['جان', 'لڑک', 'میدان', 'کھیل']`

All three new stop words were removed.

### 3. One-hot vector for a whole sentence
With vocabulary `['cat', 'sat', 'mat', 'dog']`, adding the one-hot vectors of `cat sat mat` gives:

```
[1 1 1 0]
```

`cat`, `sat` and `mat` are present. `dog` is not.

### 4. Highest TF-IDF word in `'dog sat'`
Documents: `['cat sat cat', 'dog sat', 'cat ate']`

| Word | TF-IDF | IDF | Appears in |
|------|--------|-----|-----------|
| dog | 0.796 | 1.693 | 1 of 3 docs |
| sat | 0.605 | 1.288 | 2 of 3 docs |

**Answer: `dog`.** Both words appear once in `'dog sat'`, so their term frequency is equal. The difference comes from IDF. `sat` appears in two documents so it is common and gets a lower weight. `dog` appears in only one document so it is rare and gets a higher weight.

### 5. Urdu sentence through the pipeline
Sentence: `مجھے کرکٹ کا کھیل پسند ہے`

| Step | Result |
|------|--------|
| Normalize + tokenize | `مجھے, کرکٹ, کا, کھیل, پسند, ہے` |
| Remove stop words | `مجھے, کرکٹ, کھیل, پسند` |
| Stemming (final tokens) | `مجھ, کرکٹ, کھیل, پسند` |

Each of the four final tokens then gets a one-hot vector of length 4.

**Limitation:** the simple Urdu stemmer cut the last letter of `مجھے` and made it `مجھ`. Really, `مجھے` should be in the stop-word list. This shows how basic rule-based stemmers can make mistakes.

---

## How to run

### Option 1: Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook** and select the `.ipynb` file.
3. Run the cells from top to bottom.

### Option 2: Locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install numpy scikit-learn nltk jupyter
jupyter notebook
```

Then open `Areeha_Shafique_Text_Preprocessing_NLP_Solved.ipynb` and run all cells.

---

## Requirements

| Library | Needed for | Required? |
|---------|-----------|-----------|
| `numpy` | Vector math (exercise 3) | Yes |
| `scikit-learn` | One-hot encoder, Bag of Words, TF-IDF | Yes |
| `nltk` | Better tokenizer, stemmer and lemmatizer (needs WordNet data) | Optional for the lecture demos, needed for exercise 1 |

The lecture demos use pure-Python helpers, so most cells run without any downloads. NLTK data (`punkt`, `wordnet`, and so on) is downloaded automatically by the setup cell if the internet is available.

---

## Key things I learned

- Preprocessing is not the same for every language. English tricks like lowercasing do nothing for Urdu, and an English stop-word list removes nothing from Urdu text.
- Stemming is quick but rough. Lemmatization is slower but gives real words.
- Do not remove stop words blindly. Words like `not` can change the meaning in tasks such as sentiment analysis.
- TF-IDF gives more weight to rare, informative words and less weight to words that appear everywhere.
- One-hot, Bag of Words and TF-IDF are sparse and carry no meaning. Embeddings fix this, but they still start from clean tokens.

---

## Repository structure

```
.
├── Areeha_Shafique_Text_Preprocessing_NLP_Solved.ipynb   # Lecture demo + solved exercises
└── README.md
```

---

## Author

**Areeha Shafique**

The lecture notebook and its demos come from the course material. The solutions to the Section 12 exercises are my own work.
