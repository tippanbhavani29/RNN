import streamlit as st
import pandas as pd
from datetime import datetime
import os

from utils import predict_sentiment

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="SentimentAI",
    page_icon="🎬",
    layout="wide"
)

# ==================================
# CSS
# ==================================

st.markdown("""
<style>

.main{
background-color:#f8f9fa;
}

.sentiment-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==================================
# SIDEBAR
# ==================================

st.sidebar.title("🎬 SentimentAI")

st.sidebar.success(
    "RNN Sentiment Engine"
)

st.sidebar.markdown("""
### Model Information

Architecture:
SimpleRNN

Dataset:
IMDB

Vocabulary:
10,000 Words

Sequence Length:
500
""")

# ==================================
# HEADER
# ==================================

st.markdown("""
# 🎬 SentimentAI

### Movie Review Sentiment Analysis using RNN
""")

st.markdown("---")

# ==================================
# METRICS
# ==================================

m1,m2,m3 = st.columns(3)

m1.metric(
    "Model",
    "SimpleRNN"
)

m2.metric(
    "Vocabulary",
    "10K"
)

m3.metric(
    "Task",
    "Binary Classification"
)

st.markdown("---")

# ==================================
# TEXT INPUT
# ==================================

review = st.text_area(
    "Enter Movie Review",
    height=200,
    placeholder="""
This movie was fantastic.
The acting was excellent.
"""
)

# ==================================
# PREDICT
# ==================================

if st.button(
    "🔍 Analyze Sentiment"
):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        sentiment, confidence = (
            predict_sentiment(review)
        )

        # ==========================
        # SAVE HISTORY
        # ==========================

        prediction_data = {

            "Date":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Review":
            review[:100],

            "Sentiment":
            sentiment,

            "Confidence":
            round(confidence,2)
        }

        csv_file = (
            "sentiment_history.csv"
        )

        if os.path.exists(csv_file):

            history_df = pd.read_csv(
                csv_file
            )

            history_df = pd.concat(
                [
                    history_df,
                    pd.DataFrame(
                        [prediction_data]
                    )
                ],
                ignore_index=True
            )

        else:

            history_df = pd.DataFrame(
                [prediction_data]
            )

        history_df.to_csv(
            csv_file,
            index=False
        )

        # ==========================
        # DISPLAY RESULT
        # ==========================

        st.markdown("---")

        col1,col2 = st.columns(2)

        with col1:

            if sentiment == "Positive":

                st.success(
                    f"😊 {sentiment}"
                )

            else:

                st.error(
                    f"😞 {sentiment}"
                )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

        with col2:

            if sentiment == "Positive":

                st.info("""
The review contains
strong positive language
and favorable opinions.
""")

            else:

                st.warning("""
The review contains
negative sentiment
and unfavorable opinions.
""")

# ==================================
# ANALYTICS
# ==================================

st.markdown("---")

st.header(
    "📈 Sentiment Analytics"
)

csv_file = "sentiment_history.csv"

if os.path.exists(csv_file):

    history_df = pd.read_csv(
        csv_file
    )

    a,b,c = st.columns(3)

    a.metric(
        "Total Predictions",
        len(history_df)
    )

    positive_count = len(
        history_df[
            history_df["Sentiment"]
            == "Positive"
        ]
    )

    negative_count = len(
        history_df[
            history_df["Sentiment"]
            == "Negative"
        ]
    )

    b.metric(
        "Positive Reviews",
        positive_count
    )

    c.metric(
        "Negative Reviews",
        negative_count
    )

    st.markdown("---")

    left,right = st.columns(2)

    with left:

        st.subheader(
            "Prediction History"
        )

        st.dataframe(
            history_df.tail(10),
            use_container_width=True
        )

    with right:

        st.subheader(
            "Sentiment Distribution"
        )

        sentiment_counts = (
            history_df["Sentiment"]
            .value_counts()
        )

        st.bar_chart(
            sentiment_counts
        )

# ==================================
# FOOTER
# ==================================

st.markdown("---")

st.caption(
    "🎬 SentimentAI | TensorFlow + RNN + Streamlit"
)