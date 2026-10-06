"""Review Radar - sentiment analysis dashboard for product reviews (Streamlit)."""
import io
from collections import Counter
import altair as alt
import pandas as pd
import streamlit as st
import sentiment_analysis as sa

st.set_page_config(page_title="Review Radar", page_icon="📡", layout="wide")
COL = {"Positive": "#7fe0b8", "Neutral": "#cdb8ff", "Negative": "#ff9fbd"}
ORDER = list(COL)
SCALE = alt.Scale(domain=ORDER, range=list(COL.values()))
ICON = {"Battery & Charging": "🔋", "Sound & Calls": "🎧", "Comfort & Build": "🧱", "Price & Value": "💰", "Service & Delivery": "📦"}
SAMPLE_NAME = "Real sample: Amazon phone-accessory reviews"


def load_upload(f):
    raw = f.getvalue().decode("utf-8", errors="ignore")
    if f.name.lower().endswith(".csv"):
        d = pd.read_csv(io.StringIO(raw))
        col = next((c for c in d.columns if c.lower() in ("review", "text", "post", "comment", "sentence")), d.columns[0])
        out = pd.DataFrame({"review": d[col].astype(str)})
        if "label" in d.columns:
            out["label"] = d["label"]
        return out
    return pd.DataFrame({"review": [l.strip() for l in raw.splitlines() if l.strip()]})


@st.cache_data
def analyse(df):
    out = df.copy()
    res = [sa.score(t) for t in out.review]
    out["prediction"] = [sa.label(r[0]) for r in res]
    out["score"] = [round(r[0], 2) for r in res]
    out["topics"] = [", ".join(k for k, kw in sa.THEMES.items() if any(x in r[2] for x in kw.split())) for r in res]
    out["praise"] = [[w for w, s in r[1] if s > 0] for r in res]
    out["complaints"] = [[w for w, s in r[1] if s < 0] for r in res]
    if "label" in out:
        out["human"] = out.label.map(sa.to_label)
    return out


# ---------- sidebar ----------
st.sidebar.title("📡 Review Radar")
source = st.sidebar.radio("Which reviews should I analyse?", [SAMPLE_NAME, "Upload a file", "Paste reviews"])
if source == "Upload a file":
    f = st.sidebar.file_uploader("CSV or TXT. A CSV may have a 'label' column (1 = positive, 0 = negative)", type=["csv", "txt"])
    raw_df = load_upload(f) if f else pd.DataFrame()
elif source == "Paste reviews":
    txt = st.sidebar.text_area("One review per line", height=200)
    raw_df = pd.DataFrame({"review": [l.strip() for l in txt.splitlines() if l.strip()]})
else:
    raw_df = pd.read_csv("reviews.csv")
st.sidebar.caption("Tip: use the ⋮ menu (top right) > Settings to switch between light and dark mode.")

# ---------- header ----------
st.title("Review Radar 📡")
st.write("Reads customer reviews, classifies each one as positive, neutral or negative, and shows what customers like and complain about.")
if source == SAMPLE_NAME:
    st.caption("Data: 240 real customer sentences from Amazon, labelled positive or negative by people. "
               "From the UCI Sentiment Labelled Sentences dataset (Kotzias et al., KDD 2015).")
one = st.text_input("🧪 Try one review first", placeholder="The battery died after two days")
if one:
    total, hits, _ = sa.score(one)
    v = sa.label(total)
    st.info(f"**{v}** (score {total:.1f}) " + (f"from: {', '.join(w for w, _ in hits)}" if hits else "no sentiment words found"))
if raw_df.empty:
    st.warning("Add some reviews in the sidebar to see the dashboard.")
    st.stop()

df = analyse(raw_df)
n = len(df)
has_human = "human" in df
counts = df.prediction.value_counts().reindex(ORDER, fill_value=0)
net = (counts["Positive"] - counts["Negative"]) / n
st.header("😄 Customers are mostly happy" if net > .3 else "🙂 Leaning positive" if net > .05 else
          "😐 Mixed feelings" if net > -.05 else "🙁 Leaning negative" if net > -.3 else "😠 Mostly unhappy")
c = st.columns(6 if has_human else 5)
c[0].metric("Reviews analysed", n)
for i, k in enumerate(ORDER, 1):
    c[i].metric(k, f"{counts[k] / n:.0%}", f"{counts[k]} reviews", delta_color="off")
c[4].metric("Net sentiment", f"{net * 100:+.0f}")
if has_human:
    c[5].metric("Accuracy vs humans", f"{(df.prediction == df.human).mean():.0%}")

names = ["📊 Overview"] + (["🎯 Accuracy check"] if has_human else []) + ["🏷️ Topics & words", "💬 Reviews"]
tabs = dict(zip(names, st.tabs(names)))

with tabs["📊 Overview"]:
    share = counts.rename_axis("sentiment").reset_index(name="reviews")
    st.altair_chart(alt.Chart(share).mark_arc(innerRadius=70).encode(
        theta="reviews:Q", color=alt.Color("sentiment:N", scale=SCALE, legend=alt.Legend(title=None)),
        tooltip=["sentiment", "reviews"]).properties(height=300), use_container_width=True)
    if has_human:
        st.subheader("What the tool says vs what people said")
        cmp = pd.concat([df.prediction.value_counts().reindex(ORDER, fill_value=0).rename("Tool"),
                         df.human.value_counts().reindex(ORDER, fill_value=0).rename("Human labels")], axis=1)
        cmp = cmp.rename_axis("sentiment").reset_index().melt("sentiment", var_name="source", value_name="reviews")
        st.altair_chart(alt.Chart(cmp).mark_bar().encode(
            y=alt.Y("source:N", title=None), x=alt.X("reviews:Q", stack="zero"),
            color=alt.Color("sentiment:N", scale=SCALE, legend=alt.Legend(title=None)),
            tooltip=["source", "sentiment", "reviews"]).properties(height=140), use_container_width=True)

if has_human:
    with tabs["🎯 Accuracy check"]:
        st.write("Every review in this dataset was labelled by a person, so we can check how often the tool agrees. "
                 "The tool can also answer *Neutral*, which counts as a miss because humans only chose positive or negative.")
        if "split" in df:
            sp = df.groupby("split").apply(lambda g: pd.Series({"reviews": len(g), "accuracy": f"{(g.prediction == g.human).mean():.1%}"}),
                                           include_groups=False)
            sp.index = sp.index.map({"dev": "dev (used to improve the word list)", "test": "test (never used for tuning, the honest score)"})
            st.dataframe(sp, use_container_width=True)
        cm = (pd.crosstab(df.human, df.prediction).reindex(index=["Positive", "Negative"], columns=ORDER, fill_value=0)
              .rename_axis("human").reset_index().melt("human", var_name="prediction", value_name="reviews"))
        base = alt.Chart(cm).encode(x=alt.X("prediction:N", sort=ORDER, title="Tool says"),
                                    y=alt.Y("human:N", sort=["Positive", "Negative"], title="Humans said"))
        st.altair_chart((base.mark_rect().encode(color=alt.Color("reviews:Q", scale=alt.Scale(scheme="purples"), legend=None)) +
                         base.mark_text(fontSize=22).encode(text="reviews:Q", color=alt.condition(alt.datum.reviews > n / 6, alt.value("white"), alt.value("black")))
                         ).properties(height=240), use_container_width=True)
        pr = []
        for k in ("Positive", "Negative"):
            tp = ((df.prediction == k) & (df.human == k)).sum()
            pr.append({"class": k, "precision": f"{tp / max((df.prediction == k).sum(), 1):.0%}", "recall": f"{tp / max((df.human == k).sum(), 1):.0%}"})
        st.dataframe(pd.DataFrame(pr).set_index("class"), use_container_width=True)
        st.caption("Precision: when the tool says positive (or negative), how often is it right? Recall: of all the truly positive (or negative) reviews, how many did it find?")

with tabs["🏷️ Topics & words"]:
    t = df.assign(topic=df.topics.str.split(", ")).explode("topic")
    t = t[t.topic.notna() & (t.topic != "")]
    st.subheader("Which parts of the product do reviews mention?")
    if t.empty:
        st.write("No topic keywords found in these reviews.")
    else:
        tc = t.groupby(["topic", "prediction"]).size().reset_index(name="reviews")
        tc["topic"] = tc.topic.map(lambda x: f"{ICON.get(x, '')} {x}")
        st.altair_chart(alt.Chart(tc).mark_bar().encode(
            x="reviews:Q", y=alt.Y("topic:N", title=None), color=alt.Color("prediction:N", scale=SCALE, legend=alt.Legend(title=None)),
            tooltip=["topic", "prediction", "reviews"]).properties(height=240), use_container_width=True)
    left, right = st.columns(2)
    for col, key, title, color in [(left, "praise", "Praise words", COL["Positive"]), (right, "complaints", "Complaint words", COL["Negative"])]:
        words = Counter(w for ws in df[key] for w in ws).most_common(8)
        col.subheader(title)
        if words:
            wd = pd.DataFrame(words, columns=["word", "times"])
            col.altair_chart(alt.Chart(wd).mark_bar(color=color).encode(
                x="times:Q", y=alt.Y("word:N", sort="-x", title=None), tooltip=["word", "times"]).properties(height=240), use_container_width=True)
        else:
            col.write("None found.")

with tabs["💬 Reviews"]:
    pick = st.multiselect("Show reviews the tool rated", ORDER, default=ORDER)
    wrong = st.checkbox("Only show reviews where the tool disagrees with the human label") if has_human else False
    view = df[df.prediction.isin(pick)]
    if wrong:
        view = view[view.prediction != view.human]
    cols = ["review", "prediction", "score", "topics"] + (["human"] if has_human else [])
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download results CSV", view[cols].to_csv(index=False), "results.csv", "text/csv")

with st.expander("🤖 Get AI-written insights (copy this prompt into Claude or Gemini)"):
    tl = "; ".join(f"{k}: {int((t.topic == k).sum())} reviews" for k in sa.THEMES) if not t.empty else "no topics"
    st.code(f"I ran rule-based sentiment analysis on {n} customer reviews of phone accessories. Overall: {counts['Positive']} positive, "
            f"{counts['Neutral']} neutral, {counts['Negative']} negative. Topics mentioned: {tl}. Acting as a data analyst: "
            "1) summarise the key findings, 2) name the biggest customer pain point, 3) suggest 3 actions a seller could take, "
            "4) list the limitations of a word-list method.", language=None)
