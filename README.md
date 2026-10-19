# Review Radar 📡 – Sentiment Analysis Dashboard

A Python dashboard that reads real Amazon phone-accessory reviews, classifies each as positive, neutral or negative, shows what customers like and complain about, and checks its own accuracy against human labels.

**[🔗 Live demo](https://review-radar-dashb.streamlit.app/)**

![Review Radar dashboard](docs/dashboard.png)

*Individual project, CAPACITI Programme (Week 3: AI for Data Analysis & Insights).*

## Key features

- Rule-based sentiment engine with negation, intensifier and "but/however" handling
- Topic tagging: Battery & Charging, Sound & Calls, Comfort & Build, Price & Value, Service & Delivery
- Accuracy check with confusion matrix, precision and recall
- Upload your own CSV/TXT file, or paste reviews directly
- Machine-learning baseline (TF-IDF + logistic regression) for comparison

## Tech stack

Python · Streamlit · Pandas · Altair · scikit-learn (baseline only)

## Run it locally

1. Install Python 3.10 or newer.
2. Clone the repo and install the dependencies:
   ```bash
   git clone https://github.com/AdriyellLopis/review-radar-dashboard.git
   cd review-radar-dashboard
   pip install -r requirements.txt
   ```
3. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
   It opens at `http://localhost:8501`.
4. Optional: to run the ML baseline, `pip install scikit-learn` then `python ml_baseline.py`.

A CSV with a `label` column (1 or 0) also gets the accuracy check.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit dashboard |
| `sentiment_analysis.py` | The sentiment engine (also runs standalone and prints accuracy) |
| `ml_baseline.py` | ML comparison: TF-IDF + logistic regression |
| `reviews.csv` | 240 labelled reviews (`review`, `label`, `split`) |
| `requirements.txt` | streamlit, pandas, altair |
| `.streamlit/config.toml` | Theme colour |

## The data

- **Source:** UCI Machine Learning Repository, *Sentiment Labelled Sentences*, Amazon subset. Created by Kotzias, Denil, De Freitas and Smyth for *From Group to Individual Labels using Deep Features* (KDD 2015). Cite the paper if you reuse it.
- **What it is:** single sentences from real Amazon reviews of phone accessories, labelled positive (1) or negative (0) by people.
- **What I used:** the first 240 of 1,000 sentences (123 positive, 117 negative). The first 120 are the **dev** set, used to improve the word list. The next 120 are the **test** set, never used for tuning.

## How it works

**Approach:** rule-based (lexicon) sentiment analysis, a basic form of text classification.

1. **Tokenise:** lowercase the review and split into words.
2. **Score words:** +1 per positive word, -1 per negative word.
3. **Handle context:** a negator within three words before flips and softens the score to ±0.8. Intensifiers multiply it. Words before *but* or *however* count for less. A short phrase list (*don't waste*, *must have*) adds its own score.
4. **Classify:** above +0.5 is Positive, below -0.5 is Negative, otherwise Neutral.
5. **Tag topics:** keyword lists assign each review to a topic.
6. **Evaluate:** compare predictions with human labels.

**Why this method:** every label can be traced to the words behind it, it needs no training data, and it runs instantly. It is a clear baseline before moving to a trained model or an LLM.

**Overfitting check:** accuracy on the dev set rose from 70.8% to 84.2% after tuning, but on the untouched test set only from 74.2% to 75.8%. Tuning on a small set improves that set far more than new data.

## Results

| Measure | Result |
| --- | --- |
| Accuracy on test set (never tuned on) | **75.8%** |
| Accuracy on dev set (used for tuning) | 84.2% |
| Accuracy on all 240 reviews | 80.0% |
| ML baseline on test set | 77.5% |
| Accuracy when the tool commits (not Neutral) | 93.7%, on 85% of reviews |

| Humans said → Tool said | Negative | Neutral | Positive |
| --- | --- | --- | --- |
| Negative (117) | 78 | 27 | 12 |
| Positive (123) | 1 | 8 | 114 |

- **Precision:** right 91% of the time when it says positive, 99% when it says negative.
- **Recall:** finds 93% of positive reviews but only 67% of negative ones. The 27 negative reviews it called Neutral are the main weakness.

## Data insights (240 real Amazon reviews)

**Headline:** reviews split about evenly between happy and unhappy customers, and battery and charging is the weakest area.

| Topic | Reviews mentioning it | Share labelled negative by people | Tool: positive / neutral / negative |
| --- | --- | --- | --- |
| 🔋 Battery & Charging | 20 | 75% | 6 / 5 / 9 |
| 🎧 Sound & Calls | 28 | 61% | 13 / 4 / 11 |
| 🧱 Comfort & Build | 26 | 42% | 15 / 5 / 6 |
| 💰 Price & Value | 15 | 40% | 9 / 0 / 6 |
| 📦 Service & Delivery | 9 | 33% | 6 / 0 / 3 |

1. **Battery and charging is the biggest pain point:** 15 of 20 mentions were negative (dying batteries, chargers that don't work).
2. **Sound and calls is second weakest:** 17 of 28 negative (reception, static, caller can't be heard).
3. **Comfort, build and price are mixed:** about 4 in 10 negative.
4. **Service and delivery looks healthy,** but only 9 reviews mention it.
5. **Praise is generic, complaints are specific:** top praise words are *great*, *good*, *works*; top complaints are *poor*, *waste*, *disappointed*.

**Recommendations for a phone-accessory seller:** check battery and charger quality first; look into signal and sound complaints and state phone compatibility clearly; read Neutral reviews by hand because many hide real problems; feature what customers praise (comfort, sturdiness, value).

## Limitations

- **Neutral is a built-in miss:** the dataset has only positive and negative labels, so a Neutral answer counts as wrong.
- **Complaints without mood words** (e.g. "only play music in one ear") are rated Neutral.
- **Small, older data:** 240 sentences, mostly older phone models.
- **Sentences, not whole reviews:** no wider context, no sarcasm detection.

## Next steps

Run on the full 1,000 sentences (or the Yelp and IMDB files), expand the word lists, try VADER or a pretrained transformer, and compare against an LLM classifier.

## Author

Adriyell Lopis · [Portfolio](https://adriyell-portfolio-3o96.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/adriyell-lopis-563921337)
