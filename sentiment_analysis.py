"""Review Radar - rule-based sentiment engine for product reviews.
Usage: python sentiment_analysis.py [reviews.csv]  (prints an evaluation if the file has a label column)"""
import re, sys
from collections import Counter

POS = set("""good great excellent love loved loves like liked best better nice perfect happy pleased satisfied recommend recommended
works worked work fine comfortable comfortably easy sturdy durable clear clearly fast quickly impressed impressive awesome amazing
fantastic wonderful superb incredible incredibly brilliant beautiful sharp solid reliable helpful value bargain lightweight smooth flawless
flawlessly cool decent glad enjoy enjoyed favorite winner tremendous fabulous worth secure stylish classy adorable super peachy
elegant protects protection handy useful ideal pretty exactly seamlessly crisp loud well simple simpler working holds hold""".split())
NEG = set("""bad poor terrible awful horrible worst waste wasted wasting worthless useless junk crap garbage disappointed disappointing
disappointment disappoint broke broken breaks break breakage died dead failed fails fail problem problems defective flimsy hate hated
unacceptable unacceptible unusable uncomfortable annoying frustrating static distorted muffled weak slow difficult refuse refused return
returned returning sucks sucked complaint issue issues unreliable avoid worse regret misleading overpriced expensive freezes freeze frozen
lousy ugly ridiculous pathetic embarrassing painful loose tinny defect flaw lost drop drops dropped dropping stupid mistake foolish
unhappy upset wrong wrongly poorly misleading deaf garbled fooled beware forget drawback trouble supposedly apparently unless""".split())
NEGATORS = {"not", "no", "never", "nothing", "hardly", "cannot", "doesn't", "didn't", "don't", "can't", "couldn't", "won't",
            "wouldn't", "isn't", "wasn't", "aren't", "weren't", "wont", "dont", "doesnt", "didnt", "couldnt", "without"}
BOOST = {"very": 1.5, "really": 1.5, "extremely": 2.0, "so": 1.3, "absolutely": 1.5, "highly": 1.5, "too": 1.2}
PHRASES = {"don't waste": -1.5, "dont waste": -1.5, "do not buy": -1.5, "don't buy": -1.5, "dont buy": -1.5, "forget about it": -1.0,
           "no longer": -1.0, "too big": -1.0, "runs down": -1.5, "must have": 1.5, "order again": 1.5, "buy again": 1.5,
           "order from them again": 1.5, "no way": -1.0, "money back": -1.0}
CONTRAST = {"but", "however"}
THEMES = {
    "Battery & Charging": "battery batteries charge charger charging charged power plug",
    "Sound & Calls": "sound audio volume mic microphone speaker call calls hear reception signal static voice",
    "Comfort & Build": "comfortable comfortably fit fits ear case plastic sturdy durable broke design buttons clip leather screen",
    "Price & Value": "price priced money value bargain worth cheap expensive",
    "Service & Delivery": "shipping shipped delivery arrived service seller customer support refund returned",
}

def score(text):
    words = re.findall(r"[a-z']+", text.lower())
    total, hits = 0.0, []
    for i, w in enumerate(words):
        s = 1 if w in POS else -1 if w in NEG else 0
        if not s:
            continue
        if i and words[i-1] in BOOST: s *= BOOST[words[i-1]]
        if any(x in NEGATORS for x in words[max(0, i-3):i]): s *= -0.8
        if any(c in words[i+1:] for c in CONTRAST): s *= 0.4
        total += s
        hits.append((w, s))
    low = text.lower().replace("\u2019", "'")
    for p, v in PHRASES.items():
        if p in low:
            total += v
            hits.append((p, v))
    return total, hits, words

def label(total):
    return "Positive" if total > 0.5 else "Negative" if total < -0.5 else "Neutral"

def to_label(v):
    return "Positive" if str(v).strip().lower() in ("1", "positive", "pos", "1.0") else "Negative"

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "reviews.csv")
    df["pred"] = [label(score(t)[0]) for t in df.review]
    print(df.pred.value_counts().to_dict())
    if "label" in df:
        df["truth"] = df.label.map(to_label)
        for sp, g in [("all", df)] + list(df.groupby("split")) if "split" in df else [("all", df)]:
            print(sp, "accuracy", round((g.pred == g.truth).mean(), 3), "n", len(g))
