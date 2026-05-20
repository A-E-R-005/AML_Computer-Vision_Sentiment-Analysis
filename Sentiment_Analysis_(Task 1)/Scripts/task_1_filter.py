import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score

# Data Loading

data_train = pd.read_csv('sentiment_analysis_training_data.csv')
text_train  = data_train['text'].values
labels_train = data_train['label'].values

data_val   = pd.read_csv('sentiment_analysis_validation_data.csv')
text_val   = data_val['text'].values
labels_val = data_val['label'].values

data_test  = pd.read_csv('sentiment_analysis_test_data.csv')
text_test  = data_test['text'].values

print(f"Train: {data_train.shape} | Val: {data_val.shape} | Test: {data_test.shape}")
print(f"Class 0: {(labels_train==0).sum()} | Class 1: {(labels_train==1).sum()}")

def get_confusion_matrix(true_label, pred_label):
    return confusion_matrix(true_label, pred_label)

def save_as_csv(pred_labels, location='.'):
    assert pred_labels.shape[0] == 1434, 'wrong number of labels, should be 1434'
    np.savetxt(location + '/results_task1.csv', pred_labels, delimiter=',')

# Heuristic Reference Labels
def heuristic_spam_detection(texts):
    spam_patterns = [
        'subject:', 'forwarded by', 'original message',
        '@', 'nom', 'meter', 'mmbtu', 'schedule',
        'outage', 'unsubscribe', 'click here', '\r\n',
    ]
    flags = []
    for text in texts:
        text_lower = text.lower()
        is_spam = any(pattern in text_lower for pattern in spam_patterns)
        flags.append(is_spam)
    return np.array(flags)

crlf_count = sum(1 for t in text_train if '\r\n' in t)
heuristic_flags = heuristic_spam_detection(text_train)
heuristic_flags_val = heuristic_spam_detection(text_val)
heuristic_flags_test = heuristic_spam_detection(text_test)

print("\n--- HEURISTIC: SAMPLE FLAGGED ---")
for i in np.where(heuristic_flags)[0][:3]:
    print(f"Label: {labels_train[i]}")
    print(text_train[i][:250])
    print("---")

# TF-IDF + SVD
vectorizer_full = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    min_df=2,
    sublinear_tf=True,
    ngram_range=(1, 2)
)

x_all = vectorizer_full.fit_transform(text_train)
x_all_val = vectorizer_full.transform(text_val)
x_all_test = vectorizer_full.transform(text_test)

svd = TruncatedSVD(n_components=100, random_state=42)
x_reduced = svd.fit_transform(x_all)
x_reduced_val = svd.transform(x_all_val)
x_reduced_test = svd.transform(x_all_test)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_reduced)
x_scaled_val = scaler.transform(x_reduced_val)
x_scaled_test = scaler.transform(x_reduced_test)

print(f"\nFull TF-IDF shape: {x_all.shape}")
print(f"Variance explained (100 components): "
      f"{svd.explained_variance_ratio_.sum()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(x_reduced[~heuristic_flags, 0], x_reduced[~heuristic_flags, 1],
                c='steelblue', alpha=0.2, s=3, label='Review')
axes[0].scatter(x_reduced[heuristic_flags, 0],  x_reduced[heuristic_flags, 1],
                c='red', alpha=0.4, s=5, label='Spam (heuristic)')
axes[0].set_xlabel('SVD Component 1')
axes[0].set_ylabel('SVD Component 2')
axes[0].set_title('SVD 2D: Heuristic Labels')
axes[0].legend(markerscale=3)

axes[1].scatter(x_reduced[labels_train==0, 0], x_reduced[labels_train==0, 1],
                c='orange', alpha=0.2, s=3, label='Negative')
axes[1].scatter(x_reduced[labels_train==1, 0], x_reduced[labels_train==1, 1],
                c='green', alpha=0.2, s=3, label='Positive')
axes[1].set_xlabel('SVD Component 1')
axes[1].set_ylabel('SVD Component 2')
axes[1].set_title('SVD 2D: Sentiment Labels')
axes[1].legend(markerscale=3)

plt.suptitle('LSA/SVD Visualisation (TF-IDF on full vocabulary)', fontsize=13)
plt.tight_layout()
plt.savefig('svd_visualisation.png', dpi=150)
plt.show()

# Choosing K - Elbow Method
inertias = []
k_range = range(2,10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(x_scaled)
    inertias.append(km.inertia_)
    print(f" k={k}: inertia={km.inertia_:.1f}")

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method: Choosing Optimal k')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('kmeans_elbow.png', dpi=150)
plt.show()

# KMeans Clustering k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
kmeans.fit(x_scaled)
cluster_labels = kmeans.labels_

cluster0_spam_rate = heuristic_flags[cluster_labels == 0].mean()
cluster1_spam_rate = heuristic_flags[cluster_labels == 1].mean()
spam_cluster = 0 if cluster0_spam_rate > cluster1_spam_rate else 1

kmeans_flags = cluster_labels == spam_cluster
kmeans_flags_val = kmeans.predict(x_scaled_val) == spam_cluster
kmeans_flags_test = kmeans.predict(x_scaled_test) == spam_cluster

print(f"\nK-Means spam cluster: {spam_cluster}")
print(f"K-Means train: {kmeans_flags.sum()} "
      f"({kmeans_flags.mean()*100:.1f}%)")
print(f"K-Means val:   {kmeans_flags_val.sum()} "
      f"({kmeans_flags_val.mean()*100:.1f}%)")
print(f"K-Means test:  {kmeans_flags_test.sum()} "
      f"({kmeans_flags_test.mean()*100:.1f}%)")

# Heuristics vs K-Means Clustering
precision = precision_score(heuristic_flags, kmeans_flags)
recall = recall_score(heuristic_flags, kmeans_flags)
f1 = f1_score(heuristic_flags, kmeans_flags)
agreement = (kmeans_flags == heuristic_flags).mean()
km_catches = (kmeans_flags & heuristic_flags).sum()
km_misses = (heuristic_flags & ~kmeans_flags).sum()
km_extras = (kmeans_flags & ~heuristic_flags).sum()

print(f"\n--- K-Means vs Heuristic ---")
print(f"Agreement:  {agreement*100:.1f}%")
print(f"Precision:  {precision:.3f}")
print(f"Recall:     {recall:.3f}")
print(f"F1:         {f1:.3f}")
print(f"Catches:    {km_catches}")
print(f"Misses:     {km_misses}")
print(f"Extras:     {km_extras}")


# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(x_reduced[~kmeans_flags, 0], x_reduced[~kmeans_flags, 1],
                c='steelblue', alpha=0.2, s=3, label='Review')
axes[0].scatter(x_reduced[kmeans_flags, 0],  x_reduced[kmeans_flags, 1],
                c='red', alpha=0.4, s=5, label='Spam (K-Means)')
axes[0].set_xlabel('SVD Component 1')
axes[0].set_ylabel('SVD Component 2')
axes[0].set_title('K-Means Predictions')
axes[0].legend(markerscale=3)

axes[1].scatter(x_reduced[~heuristic_flags, 0], x_reduced[~heuristic_flags, 1],
                c='steelblue', alpha=0.2, s=3, label='Review')
axes[1].scatter(x_reduced[heuristic_flags, 0],  x_reduced[heuristic_flags, 1],
                c='red', alpha=0.4, s=5, label='Spam (Heuristic)')
axes[1].set_xlabel('SVD Component 1')
axes[1].set_ylabel('SVD Component 2')
axes[1].set_title('Heuristic Reference')
axes[1].legend(markerscale=3)

agree   = kmeans_flags == heuristic_flags
axes[2].scatter(x_reduced[agree, 0],   x_reduced[agree, 1],
                c='steelblue', alpha=0.2, s=3, label='Agreement')
axes[2].scatter(x_reduced[~agree, 0],  x_reduced[~agree, 1],
                c='orange', alpha=0.6, s=8, label='Disagreement')
axes[2].set_xlabel('SVD Component 1')
axes[2].set_ylabel('SVD Component 2')
axes[2].set_title('Disagreements (K-Means vs Heuristic)')
axes[2].legend(markerscale=3)

plt.suptitle('K-Means Spam Detection in SVD Space', fontsize=13)
plt.tight_layout()
plt.savefig('kmeans_spam_detection.png', dpi=150)
plt.show()

# Heuristic + K-Means
final_spam_flags      = heuristic_flags      | kmeans_flags
final_spam_flags_val  = heuristic_flags_val  | kmeans_flags_val
final_spam_flags_test = heuristic_flags_test | kmeans_flags_test

text_train_clean   = text_train[~final_spam_flags]
labels_train_clean = labels_train[~final_spam_flags]
text_val_clean     = text_val[~final_spam_flags_val]
labels_val_clean   = labels_val[~final_spam_flags_val]
text_test_clean    = text_test[~final_spam_flags_test]
test_spam_indices  = np.where(final_spam_flags_test)[0]
test_clean_indices = np.where(~final_spam_flags_test)[0]

neg_orig  = (labels_train == 0).sum()
pos_orig  = (labels_train == 1).sum()
neg_clean = (labels_train_clean == 0).sum()
pos_clean = (labels_train_clean == 1).sum()

print(f"\nFinal union train: {final_spam_flags.sum()} "
      f"({final_spam_flags.mean()*100:.1f}%)")
print(f"Final union val:   {final_spam_flags_val.sum()} "
      f"({final_spam_flags_val.mean()*100:.1f}%)")
print(f"Final union test:  {final_spam_flags_test.sum()} "
      f"({final_spam_flags_test.mean()*100:.1f}%)")
print(f"\nTrain: {len(text_train)} → {len(text_train_clean)} "
      f"(removed {final_spam_flags.sum()})")
print(f"Val:   {len(text_val)} → {len(text_val_clean)} "
      f"(removed {final_spam_flags_val.sum()})")
print(f"Test spam flagged: {final_spam_flags_test.sum()} "
      f"(will receive dummy label)")
print(f"\nClass balance before: {neg_orig} neg / {pos_orig} pos")
print(f"Class balance after:  {neg_clean} neg / {pos_clean} pos")
print(f"Removal ratio neg/pos: "
      f"{(neg_orig-neg_clean)/max((pos_orig-pos_clean),1):.2f}")

# TFIDF Vectorization on clean data
vectorizer_final = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=2,
    sublinear_tf=True
)

vectorizer_final.fit(text_train_clean)
x_train_final = vectorizer_final.transform(text_train_clean)
x_val_final   = vectorizer_final.transform(text_val_clean)
x_test_final  = vectorizer_final.transform(text_test_clean)

print(f"\nFinal train: {x_train_final.shape}")
print(f"Final val:   {x_val_final.shape}")
print(f"Final test:  {x_test_final.shape}")

