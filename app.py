import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Product Taxonomy using Hierarchical Clustering")

st.title("🛒 Hierarchical Clustering for Product Taxonomy")
st.write("This web app groups Flipkart products into clusters using Hierarchical Clustering.")

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Flipkart Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    df['product_name'] = df['product_name'].fillna("")
    df['description'] = df['description'].fillna("")
    df['text'] = df['product_name'] + " " + df['description']

    # --------------------------------------------------
    # TF-IDF Vectorization
    # --------------------------------------------------
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df['text']).toarray()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------
    st.subheader("⚙️ Clustering Controls")
    num_clusters = st.slider("Select Number of Clusters", 2, 10, 5)
    run = st.button("🚀 Run Clustering")

    # --------------------------------------------------
    # Run Clustering
    # --------------------------------------------------
    if run:
        with st.spinner("Running Hierarchical Clustering..."):

            # Hierarchical Clustering
            linkage_matrix = linkage(X_scaled, method='ward')
            clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
            df['Cluster'] = clusters

        st.success("Clustering completed successfully!")

        # --------------------------------------------------
        # Dendrogram
        # --------------------------------------------------
        st.subheader("🌳 Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
        st.pyplot(fig)

        # --------------------------------------------------
        # Evaluation Metrics
        # --------------------------------------------------
        st.subheader("📊 Evaluation Metrics")

        df['true_category'] = df['product_category_tree'].apply(
            lambda x: x.split('>>')[0].replace('[', '').replace('"', '').strip()
        )

        le = LabelEncoder()
        true_labels = le.fit_transform(df['true_category'])

        accuracy = accuracy_score(true_labels, clusters)
        precision = precision_score(true_labels, clusters, average='weighted')
        recall = recall_score(true_labels, clusters, average='weighted')
        f1 = f1_score(true_labels, clusters, average='weighted')
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        silhouette = silhouette_score(X_scaled, clusters)

        metrics_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Adjusted Rand Index",
                "Normalized Mutual Information",
                "Silhouette Score"
            ],
            "Value": [
                accuracy,
                precision,
                recall,
                f1,
                ari,
                nmi,
                silhouette
            ]
        })

        st.table(metrics_df)

        # --------------------------------------------------
        # Clustered Products
        # --------------------------------------------------
        st.subheader("📦 Clustered Products")
        st.dataframe(df[['product_name', 'true_category', 'Cluster']].head(20))

        # --------------------------------------------------
        # Download Results
        # --------------------------------------------------
        st.download_button(
            "⬇️ Download Clustered Dataset",
            df.to_csv(index=False),
            file_name="product_taxonomy_clusters.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload the Flipkart CSV dataset to continue.")
