# Review Radar 📡 – Sentiment Analysis Dashboard for Phone-Accessory Reviews

**Sentiment Analysis Tool (Individual Project).** A Python dashboard built with Streamlit that reads real customer reviews, classifies each one as positive, neutral or negative, shows what customers like and complain about, and checks its own accuracy against human labels. It is a data dashboard, not a website: Python does the analysis and Streamlit draws the dashboard.

## The data (real, not made up)
- **Source:** UCI Machine Learning Repository, *Sentiment Labelled Sentences* dataset, Amazon subset. Created by Kotzias, Denil, De Freitas and Smyth for the paper *From Group to Individual Labels using Deep Features* (KDD 2015). Cite the paper if you reuse it.
- **What it is:** single sentences taken from real Amazon reviews of mobile-phone accessories (headsets, chargers, cases, batteries and phones). People labelled each sentence positive (1) or negative (0).
- **What I used:** the first 240 sentences of the 1,000 in the Amazon file (123 positive, 117 negative), saved in `reviews.csv`. The first 120 are the **dev** set, used to improve the word list. The next 120 are the **test** set, never used for tuning, which gives the honest accuracy.

## Files
| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard. |
| `sentiment_analysis.py` | The sentiment engine (also runs on its own and prints accuracy). |
| `ml_baseline.py` | A machine-learning comparison (TF-IDF + logistic regression). Needs `pip install scikit-learn`. |
| `reviews.csv` | The 240 labelled reviews (`review`, `label`, `split`). |
| `requirements.txt` | Libraries for the dashboard: streamlit, pandas, altair. |
| `.streamlit/config.toml` | Theme colour. |

## Run it on your computer
1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH").
2. In a terminal in this folder run `pip install -r requirements.txt`.
3. Run `streamlit run app.py`. The dashboard opens in your browser at `localhost:8501`.
4. Use the sidebar to analyse the real sample, upload your own CSV or TXT file, or paste reviews. A CSV with a `label` column (1 or 0) also gets the accuracy check.

## Deploy it (Streamlit Community Cloud, free)
Vercel cannot run Streamlit, so use Streamlit Community Cloud.
1. Create a public GitHub repo (for example `review-radar-dashboard`) and upload all the files, keeping `app.py` at the top level. Drag the `.streamlit` folder in too.
2. Go to share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main` and main file `app.py`, then **Deploy**.
4. After a couple of minutes you get a link ending in `.streamlit.app`.

## Technical explanation

**Approach.** Rule-based (lexicon) sentiment analysis, a basic form of text classification.
1. **Tokenise:** lowercase the review and split it into words.
2. **Score words:** +1 for each word in a positive word list (*great, works, comfortable*), -1 for each in a negative list (*poor, waste, broke*).
3. **Handle context:** a negator (*not, doesn't, never*) within the three words before flips the score and softens it to ±0.8. Intensifiers (*very, really, too*) multiply it. Words before *but* or *however* count for less, because the point usually comes after. A short list of phrases (*don't waste, no longer, must have*) adds its own score.
4. **Classify:** a total above +0.5 is Positive, below -0.5 is Negative, anything else is Neutral.
5. **Tag topics:** keyword lists assign each review to Battery & Charging, Sound & Calls, Comfort & Build, Price & Value or Service & Delivery.
6. **Evaluate:** compare each prediction with the human label using accuracy, a confusion matrix, precision and recall.

**Basic machine-learning ideas used.**
- **Train and test split.** I improved the word list only by looking at mistakes on the 120 dev reviews. Accuracy on the dev set rose from 70.8% to 84.2%, but on the untouched test set it went only from 74.2% to 75.8%. That gap shows *overfitting*: tuning on a small set improves that set far more than new data.
- **Baseline comparison.** A trained model (TF-IDF features + logistic regression, `ml_baseline.py`) learned from the dev reviews and scored **77.5%** on the test set, close to the rule-based tool at 75.8%.
- **Precision and recall.** Explained in the insights report below.

**Why this method.** Every label can be traced to the words behind it, it needs no training data, and it runs instantly. It is a clear baseline before moving to a trained model or an LLM.

**Limitations.**
- **Neutral is a built-in miss.** The dataset has only positive and negative labels, and its authors picked sentences with clear sentiment. When the tool answers Neutral it counts as wrong. Real reviews include many truly neutral ones.
- **Complaints without mood words.** "The ear buds only play music in one ear" and "Items stated as included ... ARE NOT INCLUDED" describe real problems without a negative word, so the tool rates both Neutral.
- **Small, older data.** Only 240 sentences, mostly about older phone models. The results may not hold for other products or for current phones.
- **Sentences, not whole reviews,** so context from the rest of the review is missing. It cannot detect sarcasm and it misses words not in the lists.

**Next steps.** Run the tool on the full 1,000 Amazon sentences (or the Yelp and IMDB files), add more word-list entries, try VADER or a pretrained transformer, and ask an LLM to classify the same reviews to compare. This links to the *Generative AI with Large Language Models* course.

## Data insights report (240 real Amazon reviews)

**Headline:** reviews are split about evenly between happy and unhappy customers, and battery and charging is the weakest area.

**1. What customers feel.** People labelled 123 reviews positive (51%) and 117 negative (49%). The tool read 126 as positive (53%), 79 as negative (33%) and 35 as neutral (15%). It under-counts negatives because many complaints do not use a negative word.

| Topic | Reviews mentioning it | Share labelled negative by people | Tool: positive / neutral / negative |
|---|---|---|---|
| 🔋 Battery & Charging | 20 | 75% | 6 / 5 / 9 |
| 🎧 Sound & Calls | 28 | 61% | 13 / 4 / 11 |
| 🧱 Comfort & Build | 26 | 42% | 15 / 5 / 6 |
| 💰 Price & Value | 15 | 40% | 9 / 0 / 6 |
| 📦 Service & Delivery | 9 | 33% | 6 / 0 / 3 |

**2. Key findings**
1. **Battery and charging is the biggest pain point.** 15 of the 20 reviews that mention it were labelled negative by people (75%). Typical issues are batteries that die quickly, chargers that do not work, and a phone that does not hold charge.
2. **Sound and calls is the second weakest area.** 17 of 28 (61%) are negative, with complaints about reception, static and people not being able to hear the caller.
3. **Comfort, build and price are mixed.** About 4 in 10 of these reviews are negative. Customers praise comfortable, sturdy items that are good value, and complain about parts that break and poor fit.
4. **Service and delivery looks healthy but the sample is small.** Only 9 reviews mention it, and two-thirds are positive, so treat it as a hint.
5. **Praise is generic, complaints are specific.** Top praise words are *great* (35 times), *good* (14), *works* (11), *like* (11) and *nice* (10). Top complaint words are *poor* (6), *waste* (5), *disappointed* (5), negated *work* as in "does not work" (5) and *problems* (4).

**3. How well the tool works**

| Measure | Result |
|---|---|
| Accuracy on test set (never tuned on) | **75.8%** |
| Accuracy on dev set (used for tuning) | 84.2% |
| Accuracy on all 240 reviews | 80.0% |
| ML baseline on test set | 77.5% |
| Accuracy when the tool commits (not Neutral) | 93.7%, on 85% of reviews |

| Humans said → Tool said | Negative | Neutral | Positive |
|---|---|---|---|
| Negative (117) | 78 | 27 | 12 |
| Positive (123) | 1 | 8 | 114 |

- **Precision:** when the tool says positive it is right 91% of the time, and when it says negative it is right 99% of the time.
- **Recall:** it finds 93% of the positive reviews but only 67% of the negative ones. The 27 negative reviews it called Neutral are the main weakness.
- **Reading:** the tool is trustworthy when it takes a side, but it misses quiet complaints. For a seller this matters, because those are the reviews to fix.

**4. Recommendations for a phone-accessory seller**
1. Check battery and charger quality first and review suppliers, because that is where unhappy customers cluster.
2. Look into signal and sound complaints, and state clearly which phones each headset is compatible with.
3. Read the Neutral reviews by hand. Many hide real problems, such as items that do not fit or are missing from the box.
4. Keep what customers praise, namely comfort, sturdiness and value for money, and feature it in listings.

**Caveat.** This is a 240-sentence sample of older product reviews. Re-run the dashboard on the full dataset or on your own reviews before drawing business conclusions.
