"""Machine-learning baseline: TF-IDF + logistic regression, trained on the 'dev' reviews and tested on the 'test' reviews."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sentiment_analysis import to_label

df = pd.read_csv("reviews.csv")
train, test = df[df.split == "dev"], df[df.split == "test"]
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1), LogisticRegression(max_iter=1000))
model.fit(train.review, train.label.map(to_label))
acc = (model.predict(test.review) == test.label.map(to_label)).mean()
print(f"ML baseline (TF-IDF + logistic regression) test accuracy: {acc:.1%} on {len(test)} reviews")
