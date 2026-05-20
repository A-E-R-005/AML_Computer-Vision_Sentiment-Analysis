import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import TruncatedSVD

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


def print_text(text, label):
    print(text, f'\nlabel == {label}')

def get_confusion_matrix(true_label, pred_label):
    return confusion_matrix(true_label, pred_label)

def save_as_csv(pred_labels, location='.'):
    assert pred_labels.shape[0] == 1434, 'wrong number of labels, should be 1434'
    np.savetxt(location + '/results_task1.csv', pred_labels, delimiter=',')

# Heuristics

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
heuristic_flags      = heuristic_spam_detection(text_train)
heuristic_flags_val  = heuristic_spam_detection(text_val)
heuristic_flags_test = heuristic_spam_detection(text_test)

print(f"Texts with \\r\\n: {crlf_count}")
print(f"Heuristic train: {heuristic_flags.sum()} ({heuristic_flags.mean()*100:.1f}%)")
print(f"Heuristic val:   {heuristic_flags_val.sum()} ({heuristic_flags_val.mean()*100:.1f}%)")
print(f"Heuristic test:  {heuristic_flags_test.sum()} ({heuristic_flags_test.mean()*100:.1f}%)")

print("\n--- HEURISTIC: SAMPLE FLAGGED ---")
for i in np.where(heuristic_flags)[0][:3]:
    print(f"Label: {labels_train[i]}")
    print(text_train[i][:250])
    print("---")


# TF-IDF Vectorisation

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=2,
    sublinear_tf=True
)

vectorizer.fit(text_train[~heuristic_flags])
x_train_tfidf = vectorizer.transform(text_train)
x_val_tfidf   = vectorizer.transform(text_val)
x_test_tfidf  = vectorizer.transform(text_test)
x_train_norm  = normalize(x_train_tfidf)
x_val_norm    = normalize(x_val_tfidf)
x_test_norm   = normalize(x_test_tfidf)

print(f"TF-IDF shape: {x_train_tfidf.shape}")


# Isolation Forest

print("--- Contamination sweep ---")
for contamination in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]:
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    preds = iso.fit_predict(x_train_tfidf)
    n_spam = (preds == -1).sum()
    print(f"  contamination={contamination}: {n_spam} flagged ({n_spam/len(text_train)*100:.1f}%)")

iso_forest = IsolationForest(n_estimators=200, contamination=0.3, random_state=42)
iso_raw_preds = iso_forest.fit_predict(x_train_tfidf)
iso_flags     = iso_raw_preds == -1
iso_scores    = iso_forest.decision_function(x_train_tfidf)

print(f"\nISO (raw): {iso_flags.sum()} flagged ({iso_flags.mean()*100:.1f}%)")
print(f"Score range: {iso_scores.min():.3f} to {iso_scores.max():.3f}")
print(f"Mean flagged: {iso_scores[iso_flags].mean():.3f} | "
      f"Mean clean: {iso_scores[~iso_flags].mean():.3f}")

plt.figure(figsize=(10, 4))
plt.hist(iso_scores[~iso_flags], bins=50, alpha=0.7, color='steelblue',
         label='Predicted clean', density=True)
plt.hist(iso_scores[iso_flags],  bins=50, alpha=0.7, color='red',
         label='Predicted spam', density=True)
plt.axvline(x=0, color='black', linestyle='--', label='Decision boundary')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.title('Isolation Forest: Anomaly Score Distribution (Raw TF-IDF)')
plt.legend()
plt.tight_layout()
plt.savefig('isolation_forest_scores_raw.png', dpi=150)
plt.show()


# One-Class SVM

oc_svm = OneClassSVM(kernel='rbf', nu=0.25, gamma='scale')
oc_svm.fit(x_train_norm)
svm_raw_preds = oc_svm.predict(x_train_norm)
svm_flags     = svm_raw_preds == -1
svm_scores    = oc_svm.decision_function(x_train_norm)

print(f"SVM (raw): {svm_flags.sum()} flagged ({svm_flags.mean()*100:.1f}%)")
print(f"Score range: {svm_scores.min():.3f} to {svm_scores.max():.3f}")
print(f"Mean flagged: {svm_scores[svm_flags].mean():.3f} | Mean clean: {svm_scores[~svm_flags].mean():.3f}")

print("\n--- METHOD AGREEMENT: ISO vs SVM (raw TF-IDF) ---")
print(f"Both flagged: {(iso_flags & svm_flags).sum()}")
print(f"ISO only:     {(iso_flags & ~svm_flags).sum()}")
print(f"SVM only:     {(svm_flags & ~iso_flags).sum()}")
print(f"Neither:      {(~iso_flags & ~svm_flags).sum()}")

print("\n--- SVM ONLY (false positive check) ---")
for i in np.where(svm_flags & ~iso_flags)[0][:3]:
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

X_all      = vectorizer_full.fit_transform(text_train)
X_all_val  = vectorizer_full.transform(text_val)
X_all_test = vectorizer_full.transform(text_test)

svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced     = svd.fit_transform(X_all)
X_reduced_val = svd.transform(X_all_val)
X_reduced_test = svd.transform(X_all_test)

scaler    = StandardScaler()
X_scaled      = scaler.fit_transform(X_reduced)
X_scaled_val  = scaler.transform(X_reduced_val)
X_scaled_test = scaler.transform(X_reduced_test)

print(f"Full TF-IDF shape: {X_all.shape}")
print(f"Variance explained (100 components): "
      f"{svd.explained_variance_ratio_.sum()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(X_reduced[~heuristic_flags, 0], X_reduced[~heuristic_flags, 1],
                c='steelblue', alpha=0.2, s=3, label='Review')
axes[0].scatter(X_reduced[heuristic_flags, 0],  X_reduced[heuristic_flags, 1],
                c='red', alpha=0.4, s=5, label='Spam (heuristic)')
axes[0].set_xlabel('SVD Component 1')
axes[0].set_ylabel('SVD Component 2')
axes[0].set_title('SVD 2D: Heuristic Labels')
axes[0].legend(markerscale=3)

axes[1].scatter(X_reduced[labels_train==0, 0], X_reduced[labels_train==0, 1],
                c='orange', alpha=0.2, s=3, label='Negative')
axes[1].scatter(X_reduced[labels_train==1, 0], X_reduced[labels_train==1, 1],
                c='green', alpha=0.2, s=3, label='Positive')
axes[1].set_xlabel('SVD Component 1')
axes[1].set_ylabel('SVD Component 2')
axes[1].set_title('SVD 2D: Sentiment Labels')
axes[1].legend(markerscale=3)

plt.suptitle('LSA/SVD Visualisation (TF-IDF on full vocabulary)', fontsize=13)
plt.tight_layout()
plt.savefig('svd_visualisation.png', dpi=150)
plt.show()


# ISO + SVM + LOF (SVD FEATURES)

iso_svd = IsolationForest(n_estimators=300, contamination=0.27, random_state=42)
iso_svd_preds  = iso_svd.fit_predict(X_scaled)
iso_svd_flags  = iso_svd_preds == -1
iso_svd_scores = iso_svd.decision_function(X_scaled)

oc_svm_svd = OneClassSVM(kernel='rbf', nu=0.27, gamma='scale')
oc_svm_svd.fit(X_scaled)
svm_svd_preds  = oc_svm_svd.predict(X_scaled)
svm_svd_flags  = svm_svd_preds == -1
svm_svd_scores = oc_svm_svd.decision_function(X_scaled)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.27, novelty=False)
lof_preds  = lof.fit_predict(X_scaled)
lof_flags  = lof_preds == -1
lof_scores = lof.negative_outlier_factor_

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, scores, flags, title, xlabel in zip(
    axes,
    [iso_svd_scores, svm_svd_scores, lof_scores],
    [iso_svd_flags,  svm_svd_flags,  lof_flags],
    ['ISO (SVD)', 'One-Class SVM (SVD)', 'LOF (SVD)'],
    ['Anomaly Score', 'Decision Score', 'Outlier Score']
):
    ax.hist(scores[~heuristic_flags], bins=50, alpha=0.7,
            color='steelblue', label='Review', density=True)
    ax.hist(scores[heuristic_flags],  bins=50, alpha=0.7,
            color='red', label='Spam', density=True)
    if title != 'LOF (SVD)':
        ax.axvline(x=0, color='black', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Anomaly Score Distributions: SVD Features', fontsize=12)
plt.tight_layout()
plt.savefig('anomaly_scores_svd.png', dpi=150)
plt.show()
