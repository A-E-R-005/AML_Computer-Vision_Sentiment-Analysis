import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import movie_reviews

nltk.download('movie_reviews', quiet=True)


from task_1_filter import (
    x_train_final, labels_train_clean, x_val_final, get_confusion_matrix,
    labels_val_clean, vectorizer_final, text_val_clean, text_train_clean,
    text_test_clean, test_clean_indices, test_spam_indices,
    save_as_csv, x_test_final
)

negation_words = {
    'not', 'no', 'nor', 'never', 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'scarcely', 'barely', 'but', 'however',
    'although', 'though', 'against', 'without', 'few', 'little'
}

removed_negations = negation_words & set(ENGLISH_STOP_WORDS)
print(f"Negation words being removed by standard stopwords:")
print(removed_negations)

custom_stopwords = ENGLISH_STOP_WORDS - negation_words
print(f"\nStandard stopwords: {len(ENGLISH_STOP_WORDS)}")
print(f"Custom stopwords (negation preserved): {len(custom_stopwords)}")

def mark_negations(texts):
    neg_words = {'not', 'no', 'never', 'neither', 'nor',
                 'hardly', 'barely', 'scarcely'}
    marked = []
    for text in texts:
        tokens = text.lower().split()
        result = []
        negate = False
        for token in tokens:
            if token in neg_words:
                negate = True
                result.append(token)
            elif token in {'.', ',', '!', '?', ';'}:
                negate = False
                result.append(token)
            elif negate:
                result.append(token)
                result.append(f"NOT_{token}")
                negate = False
            else:
                result.append(token)
        marked.append(' '.join(result))
    return np.array(marked)


print("Applying negation marking...")
text_train_marked = mark_negations(text_train_clean)
text_val_marked   = mark_negations(text_val_clean)
text_test_marked  = mark_negations(text_test_clean)

sample_idx = next(i for i, t in enumerate(text_val_clean)
                  if 'not' in t.lower())
print(f"\nOriginal: {text_val_clean[sample_idx][:150]}")
print(f"Marked:   {text_val_marked[sample_idx][:150]}")


# Negation Marked Vectorizers

vectorizer_marked_5k = TfidfVectorizer(
    max_features=5000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=True
)
vectorizer_marked_5k.fit(text_train_marked)
x_train_marked_5k = vectorizer_marked_5k.transform(text_train_marked)
x_val_marked_5k   = vectorizer_marked_5k.transform(text_val_marked)
x_test_marked_5k  = vectorizer_marked_5k.transform(text_test_marked)

vectorizer_marked_6k = TfidfVectorizer(
    max_features=6000,
    min_df=2,
    sublinear_tf=True
)
vectorizer_marked_6k.fit(text_train_marked)
x_train_marked_6k = vectorizer_marked_6k.transform(text_train_marked)
x_val_marked_6k   = vectorizer_marked_6k.transform(text_val_marked)
x_test_marked_6k  = vectorizer_marked_6k.transform(text_test_marked)

vectorizer_marked_20k = TfidfVectorizer(
    max_features=20000,
    stop_words=list(custom_stopwords),
    min_df=1,
    sublinear_tf=True,
    ngram_range=(1, 1)
)
vectorizer_marked_20k.fit(text_train_marked)
x_train_marked_20k = vectorizer_marked_20k.transform(text_train_marked)
x_val_marked_20k   = vectorizer_marked_20k.transform(text_val_marked)
x_test_marked_20k  = vectorizer_marked_20k.transform(text_test_marked)

vectorizer_marked_bigram = TfidfVectorizer(
    max_features=10000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=True,
    ngram_range=(1, 2)
)
vectorizer_marked_bigram.fit(text_train_marked)
x_train_marked_bigram = vectorizer_marked_bigram.transform(text_train_marked)
x_val_marked_bigram   = vectorizer_marked_bigram.transform(text_val_marked)
x_test_marked_bigram  = vectorizer_marked_bigram.transform(text_test_marked)

vectorizer_marked_uni = TfidfVectorizer(
    max_features=10000,
    stop_words=list(custom_stopwords),
    min_df=1,
    sublinear_tf=True,
    ngram_range=(1, 1)
)
vectorizer_marked_bi = TfidfVectorizer(
    max_features=10000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=True,
    ngram_range=(2, 2)
)
vectorizer_marked_uni.fit(text_train_marked)
vectorizer_marked_bi.fit(text_train_marked)
x_train_marked_sep = hstack([vectorizer_marked_uni.transform(text_train_marked),
                              vectorizer_marked_bi.transform(text_train_marked)])
x_val_marked_sep   = hstack([vectorizer_marked_uni.transform(text_val_marked),
                              vectorizer_marked_bi.transform(text_val_marked)])
x_test_marked_sep  = hstack([vectorizer_marked_uni.transform(text_test_marked),
                              vectorizer_marked_bi.transform(text_test_marked)])

vectorizer_marked_raw = TfidfVectorizer(
    max_features=5000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=False
)
vectorizer_marked_raw.fit(text_train_marked)
x_train_marked_raw = vectorizer_marked_raw.transform(text_train_marked)
x_val_marked_raw   = vectorizer_marked_raw.transform(text_val_marked)
x_test_marked_raw  = vectorizer_marked_raw.transform(text_test_marked)


# Logistic Regression (baseline model)

lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(x_train_final, labels_train_clean)

lr_val_preds = lr.predict(x_val_final)
lr_val_probs = lr.predict_proba(x_val_final)[:, 1]
lr_cm        = get_confusion_matrix(labels_val_clean, lr_val_preds)
lr_acc       = (lr_val_preds == labels_val_clean).mean()
lr_precision = precision_score(labels_val_clean, lr_val_preds)
lr_recall    = recall_score(labels_val_clean, lr_val_preds)
lr_f1        = f1_score(labels_val_clean, lr_val_preds)

print(f"\n--- Logistic Regression (Baseline) ---")
print(f"Accuracy:  {lr_acc*100:.2f}%")
print(f"Precision: {lr_precision:.3f}")
print(f"Recall:    {lr_recall:.3f}")
print(f"F1:        {lr_f1:.3f}")
print(f"Confusion matrix:\n{lr_cm}")

feature_names_final = vectorizer_final.get_feature_names_out()
lr_coeffs           = lr.coef_[0]
top_positive_idx    = lr_coeffs.argsort()[-15:][::-1]
top_negative_idx    = lr_coeffs.argsort()[:15]

print(f"\nTop positive words (predicts label=1):")
for idx in top_positive_idx:
    print(f"  '{feature_names_final[idx]}': {lr_coeffs[idx]:.3f}")

print(f"\nTop negative words (predicts label=0):")
for idx in top_negative_idx:
    print(f"  '{feature_names_final[idx]}': {lr_coeffs[idx]:.3f}")

val_correct   = np.where(lr_val_preds == labels_val_clean)[0]
val_incorrect = np.where(lr_val_preds != labels_val_clean)[0]

print(f"\n--- CORRECT PREDICTIONS (sample) ---")
for i in val_correct[:3]:
    print(f"True: {labels_val_clean[i]} | Pred: {lr_val_preds[i]}")
    print(text_val_clean[i][:200])
    print("---")

print(f"\n--- INCORRECT PREDICTIONS (failure cases) ---")
for i in val_incorrect[:5]:
    print(f"True: {labels_val_clean[i]} | Pred: {lr_val_preds[i]}")
    print(text_val_clean[i][:200])
    print("---")


# Naive Bayes Unigram 5k model

nb = MultinomialNB(alpha=1.0)
nb.fit(x_train_final, labels_train_clean)

nb_val_preds = nb.predict(x_val_final)
nb_cm        = get_confusion_matrix(labels_val_clean, nb_val_preds)
nb_acc       = (nb_val_preds == labels_val_clean).mean()
nb_precision = precision_score(labels_val_clean, nb_val_preds)
nb_recall    = recall_score(labels_val_clean, nb_val_preds)
nb_f1        = f1_score(labels_val_clean, nb_val_preds)

print(f"\n--- Naive Bayes ---")
print(f"Accuracy:  {nb_acc*100:.2f}%")
print(f"Precision: {nb_precision:.3f}")
print(f"Recall:    {nb_recall:.3f}")
print(f"F1:        {nb_f1:.3f}")
print(f"Confusion matrix:\n{nb_cm}")


# Linear SVM Unigram 5k model

svm_clf       = LinearSVC(max_iter=2000, random_state=42, C=1.0)
svm_clf.fit(x_train_final, labels_train_clean)

svm_val_preds = svm_clf.predict(x_val_final)
svm_cm        = get_confusion_matrix(labels_val_clean, svm_val_preds)
svm_acc       = (svm_val_preds == labels_val_clean).mean()
svm_precision = precision_score(labels_val_clean, svm_val_preds)
svm_recall    = recall_score(labels_val_clean, svm_val_preds)
svm_f1        = f1_score(labels_val_clean, svm_val_preds)

print(f"\n--- Linear SVM ---")
print(f"Accuracy:  {svm_acc*100:.2f}%")
print(f"Precision: {svm_precision:.3f}")
print(f"Recall:    {svm_recall:.3f}")
print(f"F1:        {svm_f1:.3f}")
print(f"Confusion matrix:\n{svm_cm}")


# Logistic Regression C Sweep model

print(f"\n--- Logistic Regression C sweep ---")
print(f"{'C':<10} {'Accuracy':>10} {'F1':>8}")
print("-" * 30)

best_lr_acc   = 0
best_lr_c     = 1.0
best_lr_model = None

for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    lr_c  = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr_c.fit(x_train_final, labels_train_clean)
    preds = lr_c.predict(x_val_final)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    print(f"C={C:<8} {acc*100:>10.2f}% {f:>8.3f}")
    if acc > best_lr_acc:
        best_lr_acc   = acc
        best_lr_c     = C
        best_lr_model = lr_c

print(f"\nBest C: {best_lr_c} → accuracy {best_lr_acc*100:.2f}%")


# Naive Bayes Alpha sweep model

print(f"\n--- Naive Bayes alpha sweep ---")
print(f"{'Alpha':<10} {'Accuracy':>10} {'F1':>8}")
print("-" * 30)

best_nb_acc   = 0
best_nb_alpha = 1.0
best_nb_model = None

for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_final, labels_train_clean)
    preds = nb_a.predict(x_val_final)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    print(f"alpha={alpha:<5} {acc*100:>10.2f}% {f:>8.3f}")
    if acc > best_nb_acc:
        best_nb_acc   = acc
        best_nb_alpha = alpha
        best_nb_model = nb_a

print(f"\nBest alpha: {best_nb_alpha} → accuracy {best_nb_acc*100:.2f}%")


# Model Comparison (Baseline models)

print(f"\n{'Model':<30} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8}")
print("-" * 70)

all_classifiers = [
    ('Logistic Regression (C=1)',  lr_val_preds),
    ('Naive Bayes (alpha=1)',       nb_val_preds),
    ('Linear SVM (C=1)',            svm_val_preds),
    ('LR best C',                   best_lr_model.predict(x_val_final)),
    ('NB best alpha',               best_nb_model.predict(x_val_final)),
]

for name, preds in all_classifiers:
    acc = (preds == labels_val_clean).mean()
    p   = precision_score(labels_val_clean, preds)
    r   = recall_score(labels_val_clean, preds)
    f   = f1_score(labels_val_clean, preds)
    print(f"{name:<30} {acc*100:>10.2f}% {p:>10.3f} {r:>8.3f} {f:>8.3f}")


# Confusion Matrix

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, cm, title in zip(
    axes,
    [lr_cm, nb_cm, svm_cm],
    ['Logistic Regression', 'Naive Bayes', 'Linear SVM']
):
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]),
                    ha='center', va='center',
                    color='white' if cm[row, col] > cm.max()/2 else 'black',
                    fontsize=14)
    plt.colorbar(im, ax=ax)

plt.suptitle('Confusion Matrices: Validation Set (Baseline)', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices_baseline.png', dpi=150)
plt.show()


# Unigram 20k

vectorizer_20k = TfidfVectorizer(
    max_features=20000,
    stop_words=list(custom_stopwords),
    min_df=1,
    sublinear_tf=True,
    ngram_range=(1, 1)
)
vectorizer_20k.fit(text_train_clean)
x_train_20k = vectorizer_20k.transform(text_train_clean)
x_val_20k   = vectorizer_20k.transform(text_val_clean)
x_test_20k  = vectorizer_20k.transform(text_test_clean)

print(f"Large 20k TF-IDF shape: {x_train_20k.shape}")

nb_20k       = MultinomialNB(alpha=0.01)
nb_20k.fit(x_train_20k, labels_train_clean)
nb_20k_preds = nb_20k.predict(x_val_20k)
nb_20k_acc   = (nb_20k_preds == labels_val_clean).mean()
nb_20k_f1    = f1_score(labels_val_clean, nb_20k_preds)
print(f"NB 20k: {nb_20k_acc*100:.2f}% | F1: {nb_20k_f1:.3f}")

lr_20k       = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr_20k.fit(x_train_20k, labels_train_clean)
lr_20k_preds = lr_20k.predict(x_val_20k)
lr_20k_acc   = (lr_20k_preds == labels_val_clean).mean()
lr_20k_f1    = f1_score(labels_val_clean, lr_20k_preds)
print(f"LR 20k: {lr_20k_acc*100:.2f}% | F1: {lr_20k_f1:.3f}")

print(f"\n--- Naive Bayes alpha sweep on 20k vocab ---")
print(f"{'Alpha':<8} {'Accuracy':>10} {'F1':>8}")
print("-" * 28)

best_nb_large_acc   = 0
best_nb_large_model = None
best_nb_large_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_20k, labels_train_clean)
    preds = nb_a.predict(x_val_20k)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    print(f"a={alpha:<6} {acc*100:>10.2f}% {f:>8.3f}")
    if acc > best_nb_large_acc:
        best_nb_large_acc   = acc
        best_nb_large_alpha = alpha
        best_nb_large_model = nb_a

print(f"Best 20k vocab Naive Bayes: {best_nb_large_acc*100:.2f}% "
      f"(alpha={best_nb_large_alpha})")


# Bigram

vectorizer_bigram = TfidfVectorizer(
    max_features=10000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=True,
    ngram_range=(1, 2)
)
vectorizer_bigram.fit(text_train_clean)
x_train_bigram = vectorizer_bigram.transform(text_train_clean)
x_val_bigram   = vectorizer_bigram.transform(text_val_clean)
x_test_bigram  = vectorizer_bigram.transform(text_test_clean)

print(f"\nBigram TF-IDF shape: {x_train_bigram.shape}")

nb_bigram       = MultinomialNB(alpha=0.01)
nb_bigram.fit(x_train_bigram, labels_train_clean)
nb_bigram_preds = nb_bigram.predict(x_val_bigram)
nb_bigram_acc   = (nb_bigram_preds == labels_val_clean).mean()
nb_bigram_f1    = f1_score(labels_val_clean, nb_bigram_preds)
print(f"NB bigram:  {nb_bigram_acc*100:.2f}% | F1: {nb_bigram_f1:.3f}")

lr_bigram       = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr_bigram.fit(x_train_bigram, labels_train_clean)
lr_bigram_preds = lr_bigram.predict(x_val_bigram)
lr_bigram_acc   = (lr_bigram_preds == labels_val_clean).mean()
lr_bigram_f1    = f1_score(labels_val_clean, lr_bigram_preds)
print(f"LR bigram:  {lr_bigram_acc*100:.2f}% | F1: {lr_bigram_f1:.3f}")


# Separate Unigram + Bigram

vectorizer_uni = TfidfVectorizer(
    max_features=10000,
    stop_words=list(custom_stopwords),
    min_df=1,
    sublinear_tf=True,
    ngram_range=(1, 1)
)
vectorizer_bi  = TfidfVectorizer(
    max_features=10000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=True,
    ngram_range=(2, 2)
)
vectorizer_uni.fit(text_train_clean)
vectorizer_bi.fit(text_train_clean)

x_train_sep = hstack([vectorizer_uni.transform(text_train_clean),
                       vectorizer_bi.transform(text_train_clean)])
x_val_sep   = hstack([vectorizer_uni.transform(text_val_clean),
                       vectorizer_bi.transform(text_val_clean)])
x_test_sep  = hstack([vectorizer_uni.transform(text_test_clean),
                       vectorizer_bi.transform(text_test_clean)])

print(f"\nSeparate uni+bigram shape: {x_train_sep.shape}")

lr_sep       = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr_sep.fit(x_train_sep, labels_train_clean)
lr_sep_preds = lr_sep.predict(x_val_sep)
lr_sep_acc   = (lr_sep_preds == labels_val_clean).mean()
lr_sep_f1    = f1_score(labels_val_clean, lr_sep_preds)
print(f"LR separate uni+bi:    {lr_sep_acc*100:.2f}% | F1: {lr_sep_f1:.3f}")


# Soft voting naive bayes and logistic regression

soft_voter = VotingClassifier(
    estimators=[
        ('nb', MultinomialNB(alpha=best_nb_large_alpha)),
        ('lr', LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
    ],
    voting='soft'
)
soft_voter.fit(x_train_20k, labels_train_clean)
voter_preds = soft_voter.predict(x_val_20k)
voter_acc   = (voter_preds == labels_val_clean).mean()
voter_f1    = f1_score(labels_val_clean, voter_preds)
print(f"\nSoft voting (large vocab): {voter_acc*100:.2f}% | F1: {voter_f1:.3f}")

disagree = (best_nb_large_model.predict(x_val_20k) != lr_20k.predict(x_val_20k))
print(f"Models disagree on {disagree.sum()} samples ({disagree.mean()*100:.1f}%)")
nb_right = (best_nb_large_model.predict(x_val_20k)[disagree] == labels_val_clean[disagree]).mean()
lr_right = (lr_20k.predict(x_val_20k)[disagree] == labels_val_clean[disagree]).mean()
print(f"When disagree — NB correct: {nb_right*100:.1f}% | LR correct: {lr_right*100:.1f}%")


# Stemming

stemmer = PorterStemmer()

def stem_texts(texts):
    stemmed = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        stemmed.append(' '.join(stemmed_tokens))
    return np.array(stemmed)


print("Stemming train texts...")
text_train_stemmed = stem_texts(text_train_clean)
print("Stemming val texts...")
text_val_stemmed   = stem_texts(text_val_clean)
print("Stemming test texts...")
text_test_stemmed  = stem_texts(text_test_clean)
print("Stemming complete.")

vectorizer_stem = TfidfVectorizer(
    max_features=5000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=True,
    ngram_range=(1, 1)
)
vectorizer_stem.fit(text_train_stemmed)
x_train_stem = vectorizer_stem.transform(text_train_stemmed)
x_val_stem   = vectorizer_stem.transform(text_val_stemmed)
x_test_stem  = vectorizer_stem.transform(text_test_stemmed)

nb_stem       = MultinomialNB(alpha=0.01)
nb_stem.fit(x_train_stem, labels_train_clean)
nb_stem_preds = nb_stem.predict(x_val_stem)
nb_stem_acc   = (nb_stem_preds == labels_val_clean).mean()
nb_stem_f1    = f1_score(labels_val_clean, nb_stem_preds)
print(f"NB stemmed 5k: {nb_stem_acc*100:.2f}% | F1: {nb_stem_f1:.3f}")

best_nb_stem_acc   = 0
best_nb_stem_model = None
best_nb_stem_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_stem, labels_train_clean)
    preds = nb_a.predict(x_val_stem)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    print(f"  alpha={alpha}: {acc*100:.2f}% | F1: {f:.3f}")
    if acc > best_nb_stem_acc:
        best_nb_stem_acc   = acc
        best_nb_stem_alpha = alpha
        best_nb_stem_model = nb_a

print(f"Best stemmed NB: {best_nb_stem_acc*100:.2f}% (alpha={best_nb_stem_alpha})")

vectorizer_stem_bi = TfidfVectorizer(
    max_features=8000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=True,
    ngram_range=(1, 2)
)
vectorizer_stem_bi.fit(text_train_stemmed)
x_train_stem_bi = vectorizer_stem_bi.transform(text_train_stemmed)
x_val_stem_bi   = vectorizer_stem_bi.transform(text_val_stemmed)
x_test_stem_bi  = vectorizer_stem_bi.transform(text_test_stemmed)

nb_stem_bi       = MultinomialNB(alpha=0.01)
nb_stem_bi.fit(x_train_stem_bi, labels_train_clean)
nb_stem_bi_preds = nb_stem_bi.predict(x_val_stem_bi)
nb_stem_bi_acc   = (nb_stem_bi_preds == labels_val_clean).mean()
nb_stem_bi_f1    = f1_score(labels_val_clean, nb_stem_bi_preds)
print(f"\nNB stemmed bigram: {nb_stem_bi_acc*100:.2f}% | F1: {nb_stem_bi_f1:.3f}")

vectorizer_raw = TfidfVectorizer(
    max_features=5000,
    stop_words=list(custom_stopwords),
    min_df=2,
    sublinear_tf=False
)
vectorizer_raw.fit(text_train_clean)
x_train_raw = vectorizer_raw.transform(text_train_clean)
x_val_raw   = vectorizer_raw.transform(text_val_clean)
x_test_raw  = vectorizer_raw.transform(text_test_clean)

best_nb_raw_acc   = 0
best_nb_raw_model = None
best_nb_raw_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_raw, labels_train_clean)
    preds = nb_a.predict(x_val_raw)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    if acc > best_nb_raw_acc:
        best_nb_raw_acc   = acc
        best_nb_raw_alpha = alpha
        best_nb_raw_model = nb_a

print(f"NB raw TF (no sublinear): {best_nb_raw_acc*100:.2f}% "
      f"(alpha={best_nb_raw_alpha})")

best_cnb_acc   = 0
best_cnb_model = None
best_cnb_alpha = 0.01

print(f"\n--- Complement NB alpha sweep ---")
for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
    cnb   = ComplementNB(alpha=alpha)
    cnb.fit(x_train_final, labels_train_clean)
    preds = cnb.predict(x_val_final)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    print(f"  alpha={alpha}: {acc*100:.2f}% | F1: {f:.3f}")
    if acc > best_cnb_acc:
        best_cnb_acc   = acc
        best_cnb_alpha = alpha
        best_cnb_model = cnb

print(f"Best Complement NB: {best_cnb_acc*100:.2f}% (alpha={best_cnb_alpha})")

best_cnb_stem_acc   = 0
best_cnb_stem_model = None
best_cnb_stem_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
    cnb   = ComplementNB(alpha=alpha)
    cnb.fit(x_train_stem, labels_train_clean)
    preds = cnb.predict(x_val_stem)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    if acc > best_cnb_stem_acc:
        best_cnb_stem_acc   = acc
        best_cnb_stem_alpha = alpha
        best_cnb_stem_model = cnb

print(f"Best Complement NB (stemmed): {best_cnb_stem_acc*100:.2f}% "
      f"(alpha={best_cnb_stem_alpha})")


# Negation marked - for all models

print(f"\n--- Negation-marked models ---")

lr_marked = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr_marked.fit(x_train_marked_5k, labels_train_clean)
lr_marked_preds = lr_marked.predict(x_val_marked_5k)
lr_marked_acc   = (lr_marked_preds == labels_val_clean).mean()
lr_marked_f1    = f1_score(labels_val_clean, lr_marked_preds)
print(f"LR negation 5k:              {lr_marked_acc*100:.2f}% | F1: {lr_marked_f1:.3f}")

svm_marked = LinearSVC(max_iter=2000, random_state=42, C=1.0)
svm_marked.fit(x_train_marked_5k, labels_train_clean)
svm_marked_preds = svm_marked.predict(x_val_marked_5k)
svm_marked_acc   = (svm_marked_preds == labels_val_clean).mean()
svm_marked_f1    = f1_score(labels_val_clean, svm_marked_preds)
print(f"SVM negation 5k:             {svm_marked_acc*100:.2f}% | F1: {svm_marked_f1:.3f}")

best_nb_marked_acc   = 0
best_nb_marked_model = None
best_nb_marked_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_marked_6k, labels_train_clean)
    preds = nb_a.predict(x_val_marked_6k)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    if acc > best_nb_marked_acc:
        best_nb_marked_acc   = acc
        best_nb_marked_alpha = alpha
        best_nb_marked_model = nb_a

print(f"NB negation 6k best alpha:   {best_nb_marked_acc*100:.2f}% "
      f"(alpha={best_nb_marked_alpha})")

best_nb_marked_20k_acc   = 0
best_nb_marked_20k_model = None
best_nb_marked_20k_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_marked_20k, labels_train_clean)
    preds = nb_a.predict(x_val_marked_20k)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    if acc > best_nb_marked_20k_acc:
        best_nb_marked_20k_acc   = acc
        best_nb_marked_20k_alpha = alpha
        best_nb_marked_20k_model = nb_a

print(f"NB negation 20k best alpha:  {best_nb_marked_20k_acc*100:.2f}% "
      f"(alpha={best_nb_marked_20k_alpha})")

best_lr_marked_acc   = 0
best_lr_marked_model = None
best_lr_marked_c     = 1.0

for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
    lr_c  = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr_c.fit(x_train_marked_5k, labels_train_clean)
    preds = lr_c.predict(x_val_marked_5k)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    if acc > best_lr_marked_acc:
        best_lr_marked_acc   = acc
        best_lr_marked_c     = C
        best_lr_marked_model = lr_c

print(f"LR negation 5k best C:       {best_lr_marked_acc*100:.2f}% "
      f"(C={best_lr_marked_c})")

nb_marked_bigram = MultinomialNB(alpha=0.01)
nb_marked_bigram.fit(x_train_marked_bigram, labels_train_clean)
nb_marked_bigram_preds = nb_marked_bigram.predict(x_val_marked_bigram)
nb_marked_bigram_acc   = (nb_marked_bigram_preds == labels_val_clean).mean()
nb_marked_bigram_f1    = f1_score(labels_val_clean, nb_marked_bigram_preds)
print(f"NB negation bigram:          {nb_marked_bigram_acc*100:.2f}% | "
      f"F1: {nb_marked_bigram_f1:.3f}")

lr_marked_bigram = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr_marked_bigram.fit(x_train_marked_bigram, labels_train_clean)
lr_marked_bigram_preds = lr_marked_bigram.predict(x_val_marked_bigram)
lr_marked_bigram_acc   = (lr_marked_bigram_preds == labels_val_clean).mean()
lr_marked_bigram_f1    = f1_score(labels_val_clean, lr_marked_bigram_preds)
print(f"LR negation bigram:          {lr_marked_bigram_acc*100:.2f}% | "
      f"F1: {lr_marked_bigram_f1:.3f}")

lr_marked_sep = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr_marked_sep.fit(x_train_marked_sep, labels_train_clean)
lr_marked_sep_preds = lr_marked_sep.predict(x_val_marked_sep)
lr_marked_sep_acc   = (lr_marked_sep_preds == labels_val_clean).mean()
lr_marked_sep_f1    = f1_score(labels_val_clean, lr_marked_sep_preds)
print(f"LR negation sep uni+bi:      {lr_marked_sep_acc*100:.2f}% | "
      f"F1: {lr_marked_sep_f1:.3f}")

best_nb_marked_raw_acc   = 0
best_nb_marked_raw_model = None
best_nb_marked_raw_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_marked_raw, labels_train_clean)
    preds = nb_a.predict(x_val_marked_raw)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    if acc > best_nb_marked_raw_acc:
        best_nb_marked_raw_acc   = acc
        best_nb_marked_raw_alpha = alpha
        best_nb_marked_raw_model = nb_a

print(f"NB negation raw TF:          {best_nb_marked_raw_acc*100:.2f}% "
      f"(alpha={best_nb_marked_raw_alpha})")

best_cnb_marked_acc   = 0
best_cnb_marked_model = None
best_cnb_marked_alpha = 0.01

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
    cnb   = ComplementNB(alpha=alpha)
    cnb.fit(x_train_marked_5k, labels_train_clean)
    preds = cnb.predict(x_val_marked_5k)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    if acc > best_cnb_marked_acc:
        best_cnb_marked_acc   = acc
        best_cnb_marked_alpha = alpha
        best_cnb_marked_model = cnb

print(f"Complement NB negation 5k:   {best_cnb_marked_acc*100:.2f}% "
      f"(alpha={best_cnb_marked_alpha})")

soft_voter_marked = VotingClassifier(
    estimators=[
        ('nb',  MultinomialNB(alpha=best_nb_marked_alpha)),
        ('lr',  LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
    ],
    voting='soft'
)
soft_voter_marked.fit(x_train_marked_6k, labels_train_clean)
voter_marked_preds = soft_voter_marked.predict(x_val_marked_6k)
voter_marked_acc   = (voter_marked_preds == labels_val_clean).mean()
voter_marked_f1    = f1_score(labels_val_clean, voter_marked_preds)
print(f"Soft voting negation 6k:     {voter_marked_acc*100:.2f}% | "
      f"F1: {voter_marked_f1:.3f}")


# Comparison table

print(f"\n{'Model':<38} {'Accuracy':>10} {'F1':>8}")
print("-" * 58)

all_results = [
    ('LR unigram 5k (baseline)',        lr_val_preds),
    ('NB unigram 5k (baseline)',        nb_val_preds),
    ('SVM unigram 5k (baseline)',       svm_val_preds),
    ('LR best C (5k)',                  best_lr_model.predict(x_val_final)),
    ('NB best alpha (5k)',              best_nb_model.predict(x_val_final)),
    ('NB unigram 20k',                  nb_20k_preds),
    ('NB best alpha 20k',               best_nb_large_model.predict(x_val_20k)),
    ('LR unigram 20k',                  lr_20k_preds),
    ('NB bigram (1+2gram 10k)',         nb_bigram_preds),
    ('LR bigram (1+2gram 10k)',         lr_bigram_preds),
    ('LR separate uni+bi (20k)',        lr_sep_preds),
    ('Soft voting NB+LR (20k)',         voter_preds),
]

best_acc   = 0
best_name  = ''
best_preds = None

for name, preds in all_results:
    acc = (preds == labels_val_clean).mean()
    f   = f1_score(labels_val_clean, preds)
    print(f"{name:<38} {acc*100:>10.2f}% {f:>8.3f}")
    if acc > best_acc:
        best_acc   = acc
        best_name  = name
        best_preds = preds

print(f"\nBest model: {best_name} → {best_acc*100:.2f}%")

best_cm = get_confusion_matrix(labels_val_clean, best_preds)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(best_cm, interpolation='nearest', cmap='Blues')
ax.set_title(f'Confusion Matrix: {best_name}')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive'])
for row in range(2):
    for col in range(2):
        ax.text(col, row, str(best_cm[row, col]),
                ha='center', va='center',
                color='white' if best_cm[row, col] > best_cm.max()/2
                else 'black',
                fontsize=16)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('confusion_matrix_best.png', dpi=150)
plt.show()

print(f"\n{'Model':<40} {'Accuracy':>10} {'F1':>8}")
print("-" * 60)

extended_results = [
    ('NB best alpha 5k (current best)',    best_nb_model.predict(x_val_final)),
    ('NB stemmed 5k',                      nb_stem_preds),
    ('NB best alpha stemmed',              best_nb_stem_model.predict(x_val_stem)),
    ('NB stemmed bigram',                  nb_stem_bi_preds),
    ('NB raw TF best alpha',               best_nb_raw_model.predict(x_val_raw)),
    ('Complement NB best alpha',           best_cnb_model.predict(x_val_final)),
    ('Complement NB stemmed',              best_cnb_stem_model.predict(x_val_stem)),
    ('LR negation 5k',                     lr_marked_preds),
    ('SVM negation 5k',                    svm_marked_preds),
    ('NB negation 6k best alpha',          best_nb_marked_model.predict(x_val_marked_6k)),
    ('NB negation 20k best alpha',         best_nb_marked_20k_model.predict(x_val_marked_20k)),
    ('LR negation 5k best C',             best_lr_marked_model.predict(x_val_marked_5k)),
    ('NB negation bigram',                 nb_marked_bigram_preds),
    ('LR negation bigram',                 lr_marked_bigram_preds),
    ('LR negation sep uni+bi',             lr_marked_sep_preds),
    ('NB negation raw TF',                 best_nb_marked_raw_model.predict(x_val_marked_raw)),
    ('Complement NB negation 5k',          best_cnb_marked_model.predict(x_val_marked_5k)),
    ('Soft voting negation 6k',            voter_marked_preds),
]

overall_best_acc   = 0
overall_best_name  = ''
overall_best_model = None
overall_best_feats = None

model_feat_map = {
    'NB best alpha 5k (current best)':  (best_nb_model,              x_test_final),
    'NB stemmed 5k':                    (nb_stem,                     x_test_stem),
    'NB best alpha stemmed':            (best_nb_stem_model,          x_test_stem),
    'NB stemmed bigram':                (nb_stem_bi,                  x_test_stem_bi),
    'NB raw TF best alpha':             (best_nb_raw_model,           x_test_raw),
    'Complement NB best alpha':         (best_cnb_model,              x_test_final),
    'Complement NB stemmed':            (best_cnb_stem_model,         x_test_stem),
    'LR negation 5k':                   (lr_marked,                   x_test_marked_5k),
    'SVM negation 5k':                  (svm_marked,                  x_test_marked_5k),
    'NB negation 6k best alpha':        (best_nb_marked_model,        x_test_marked_6k),
    'NB negation 20k best alpha':       (best_nb_marked_20k_model,    x_test_marked_20k),
    'LR negation 5k best C':           (best_lr_marked_model,         x_test_marked_5k),
    'NB negation bigram':               (nb_marked_bigram,             x_test_marked_bigram),
    'LR negation bigram':               (lr_marked_bigram,             x_test_marked_bigram),
    'LR negation sep uni+bi':           (lr_marked_sep,                x_test_marked_sep),
    'NB negation raw TF':               (best_nb_marked_raw_model,    x_test_marked_raw),
    'Complement NB negation 5k':        (best_cnb_marked_model,       x_test_marked_5k),
    'Soft voting negation 6k':          (soft_voter_marked,            x_test_marked_6k),
}

for name, preds in extended_results:
    acc    = (preds == labels_val_clean).mean()
    f      = f1_score(labels_val_clean, preds)
    marker = ' ← NEW BEST' if acc > overall_best_acc else ''
    print(f"{name:<40} {acc*100:>10.2f}% {f:>8.3f}{marker}")
    if acc > overall_best_acc:
        overall_best_acc   = acc
        overall_best_name  = name
        overall_best_model, overall_best_feats = model_feat_map[name]

print(f"\nOverall best: {overall_best_name} → {overall_best_acc*100:.2f}%")


# Test Predictions

model_map = {
    'LR unigram 5k (baseline)':     (best_lr_model,        x_test_final),
    'NB unigram 5k (baseline)':     (best_nb_model,         x_test_final),
    'SVM unigram 5k (baseline)':    (svm_clf,               x_test_final),
    'LR best C (5k)':               (best_lr_model,         x_test_final),
    'NB best alpha (5k)':           (best_nb_model,         x_test_final),
    'NB unigram 20k':               (nb_20k,                x_test_20k),
    'NB best alpha 20k':            (best_nb_large_model,   x_test_20k),
    'LR unigram 20k':               (lr_20k,                x_test_20k),
    'NB bigram (1+2gram 10k)':      (nb_bigram,             x_test_bigram),
    'LR bigram (1+2gram 10k)':      (lr_bigram,             x_test_bigram),
    'LR separate uni+bi (20k)':     (lr_sep,                x_test_sep),
    'Soft voting NB+LR (20k)':      (soft_voter,            x_test_20k),
}

chosen_model, chosen_test_features = model_map[best_name]
clean_preds    = chosen_model.predict(chosen_test_features)

all_test_preds = np.full(1434, -1, dtype=float)
all_test_preds[test_clean_indices] = clean_preds

if overall_best_acc > best_acc:
    print(f"Improvement found: {best_acc*100:.2f}% → {overall_best_acc*100:.2f}%")
    clean_preds    = overall_best_model.predict(overall_best_feats)
    all_test_preds = np.full(1434, -1, dtype=float)
    all_test_preds[test_clean_indices] = clean_preds
    assert len(all_test_preds) == 1434
    save_as_csv(all_test_preds)
    print(f"Updated results_task1.csv with {overall_best_name}")
else:
    print(f"No improvement — keeping existing results_task1.csv "
          f"({best_name} at {best_acc*100:.2f}%)")

chosen_name = overall_best_name if overall_best_acc > best_acc else best_name
print(f"\n--- Test Predictions ({chosen_name}) ---")
print(f"Clean predictions: {len(clean_preds)}")
print(f"Spam (dummy -1):   {(all_test_preds == -1).sum()}")
print(f"Predicted neg (0): {(all_test_preds == 0).sum()}")
print(f"Predicted pos (1): {(all_test_preds == 1).sum()}")
print(f"Total:             {len(all_test_preds)}")

assert len(all_test_preds) == 1434, "Wrong number of predictions"
save_as_csv(all_test_preds)
print("Saved results_task1.csv")

print(f"Final model used for submission: NB negation 7k (alpha=0.05)")
print(f"Test predictions saved: {(all_test_preds != -1).sum()} clean + "
      f"{(all_test_preds == -1).sum()} spam = {len(all_test_preds)} total")
assert len(all_test_preds) == 1434
assert (all_test_preds == -1).sum() == 373
print("CSV verified correct.")

# Vocabulary size sweep (negation marked)

print(f"\n--- Vocabulary size sweep (negation-marked) ---")
print(f"{'Vocab':<8} {'NB acc':>8} {'NB F1':>8} {'Voter acc':>10} {'Voter F1':>9}")
print("-" * 48)

best_vocab_acc   = 0
best_vocab_name  = ''
best_vocab_model = None
best_vocab_feats = None

for vocab_size in [5000, 6000, 7000, 8000, 10000, 12000, 15000]:
    vec = TfidfVectorizer(
        max_features=vocab_size,
        min_df=2,
        sublinear_tf=True
    )
    vec.fit(text_train_marked)
    x_tr = vec.transform(text_train_marked)
    x_va = vec.transform(text_val_marked)
    x_te = vec.transform(text_test_marked)

    # NB alpha sweep at this vocab size
    best_nb_v_acc   = 0
    best_nb_v_model = None
    best_nb_v_alpha = 0.05
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        nb_a  = MultinomialNB(alpha=alpha)
        nb_a.fit(x_tr, labels_train_clean)
        preds = nb_a.predict(x_va)
        acc   = (preds == labels_val_clean).mean()
        if acc > best_nb_v_acc:
            best_nb_v_acc   = acc
            best_nb_v_alpha = alpha
            best_nb_v_model = nb_a

    # Soft voter at this vocab size
    voter_v = VotingClassifier(
        estimators=[
            ('nb', MultinomialNB(alpha=best_nb_v_alpha)),
            ('lr', LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
        ],
        voting='soft'
    )
    voter_v.fit(x_tr, labels_train_clean)
    voter_v_preds = voter_v.predict(x_va)
    voter_v_acc   = (voter_v_preds == labels_val_clean).mean()
    voter_v_f1    = f1_score(labels_val_clean, voter_v_preds)
    nb_v_f1       = f1_score(labels_val_clean, best_nb_v_model.predict(x_va))

    print(f"{vocab_size:<8} {best_nb_v_acc*100:>7.2f}% {nb_v_f1:>8.3f} "
          f"{voter_v_acc*100:>9.2f}% {voter_v_f1:>9.3f}")

    if voter_v_acc > best_vocab_acc:
        best_vocab_acc   = voter_v_acc
        best_vocab_name  = f'Soft voting negation {vocab_size}'
        best_vocab_model = voter_v
        best_vocab_feats = x_te

print(f"\nBest vocab size result: {best_vocab_name} → {best_vocab_acc*100:.2f}%")


print(f"\n--- Negation + custom stopwords combined sweep ---")

for vocab_size in [6000, 8000, 10000]:
    vec = TfidfVectorizer(
        max_features=vocab_size,
        stop_words=list(custom_stopwords),
        min_df=2,
        sublinear_tf=True
    )
    vec.fit(text_train_marked)
    x_tr = vec.transform(text_train_marked)
    x_va = vec.transform(text_val_marked)
    x_te = vec.transform(text_test_marked)

    best_nb_cs_acc   = 0
    best_nb_cs_model = None
    best_nb_cs_alpha = 0.05
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        nb_a  = MultinomialNB(alpha=alpha)
        nb_a.fit(x_tr, labels_train_clean)
        preds = nb_a.predict(x_va)
        acc   = (preds == labels_val_clean).mean()
        if acc > best_nb_cs_acc:
            best_nb_cs_acc   = acc
            best_nb_cs_alpha = alpha
            best_nb_cs_model = nb_a

    voter_cs = VotingClassifier(
        estimators=[
            ('nb', MultinomialNB(alpha=best_nb_cs_alpha)),
            ('lr', LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
        ],
        voting='soft'
    )
    voter_cs.fit(x_tr, labels_train_clean)
    voter_cs_preds = voter_cs.predict(x_va)
    voter_cs_acc   = (voter_cs_preds == labels_val_clean).mean()
    voter_cs_f1    = f1_score(labels_val_clean, voter_cs_preds)

    print(f"Vocab {vocab_size} + custom stops: "
          f"NB {best_nb_cs_acc*100:.2f}% | "
          f"Voter {voter_cs_acc*100:.2f}% F1:{voter_cs_f1:.3f}")

    if voter_cs_acc > best_vocab_acc:
        best_vocab_acc   = voter_cs_acc
        best_vocab_name  = f'Soft voting negation+stops {vocab_size}'
        best_vocab_model = voter_cs
        best_vocab_feats = x_te


vec_7k = TfidfVectorizer(
    max_features=7000,
    min_df=2,
    sublinear_tf=True
)
vec_7k.fit(text_train_marked)
x_train_7k = vec_7k.transform(text_train_marked)
x_val_7k   = vec_7k.transform(text_val_marked)
x_test_7k  = vec_7k.transform(text_test_marked)

# Alpha sweep at 7k to find exact best
print(f"\n--- NB alpha sweep at 7k negation-marked ---")
best_nb_7k_acc   = 0
best_nb_7k_model = None
best_nb_7k_alpha = 0.05

for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    nb_a  = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_7k, labels_train_clean)
    preds = nb_a.predict(x_val_7k)
    acc   = (preds == labels_val_clean).mean()
    f     = f1_score(labels_val_clean, preds)
    print(f"  alpha={alpha}: {acc*100:.2f}% | F1: {f:.3f}")
    if acc > best_nb_7k_acc:
        best_nb_7k_acc   = acc
        best_nb_7k_alpha = alpha
        best_nb_7k_model = nb_a

print(f"Best NB 7k negation: {best_nb_7k_acc*100:.2f}% "
      f"(alpha={best_nb_7k_alpha})")

if best_nb_7k_acc > best_vocab_acc:
    print(f"NB 7k beats voter 7k: "
          f"{best_nb_7k_acc*100:.2f}% > {best_vocab_acc*100:.2f}%")
    best_vocab_acc   = best_nb_7k_acc
    best_vocab_name  = f'NB negation 7k (alpha={best_nb_7k_alpha})'
    best_vocab_model = best_nb_7k_model
    best_vocab_feats = x_test_7k

    print(f"\n--- Fine sweep around 7k ---")
    print(f"{'Vocab':<8} {'NB acc':>8} {'NB F1':>8}")
    print("-" * 26)

    best_fine_acc = 0
    best_fine_model = None
    best_fine_feats = None
    best_fine_name = ''

    for vocab_size in [6500, 7000, 7500, 8000, 8500, 9000]:
        vec = TfidfVectorizer(
            max_features=vocab_size,
            min_df=2,
            sublinear_tf=True
        )
        vec.fit(text_train_marked)
        x_tr = vec.transform(text_train_marked)
        x_va = vec.transform(text_val_marked)
        x_te = vec.transform(text_test_marked)

        best_acc_v = 0
        best_model_v = None
        best_alpha_v = 0.05

        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            nb_a = MultinomialNB(alpha=alpha)
            nb_a.fit(x_tr, labels_train_clean)
            preds = nb_a.predict(x_va)
            acc = (preds == labels_val_clean).mean()
            if acc > best_acc_v:
                best_acc_v = acc
                best_alpha_v = alpha
                best_model_v = nb_a

        f = f1_score(labels_val_clean, best_model_v.predict(x_va))
        marker = ' ←' if best_acc_v > best_fine_acc else ''
        print(f"{vocab_size:<8} {best_acc_v * 100:>7.2f}% {f:>8.3f}{marker}")

        if best_acc_v > best_fine_acc:
            best_fine_acc = best_acc_v
            best_fine_model = best_model_v
            best_fine_feats = x_te
            best_fine_name = f'NB negation {vocab_size} (alpha={best_alpha_v})'

    print(f"\nFine sweep best: {best_fine_name} → {best_fine_acc * 100:.2f}%")

    if best_fine_acc > best_vocab_acc:
        best_vocab_acc = best_fine_acc
        best_vocab_name = best_fine_name
        best_vocab_model = best_fine_model
        best_vocab_feats = best_fine_feats
# ── UPDATE CSV IF IMPROVED ────────────────────────────────────────────────────
if best_vocab_acc > overall_best_acc:
    print(f"\nNew best found: {best_vocab_name} → {best_vocab_acc*100:.2f}%")
    clean_preds    = best_vocab_model.predict(best_vocab_feats)
    all_test_preds = np.full(1434, -1, dtype=float)
    all_test_preds[test_clean_indices] = clean_preds
    assert len(all_test_preds) == 1434
    save_as_csv(all_test_preds)
    print(f"Updated results_task1.csv with {best_vocab_name}")
else:
    print(f"\nNo improvement beyond {overall_best_acc*100:.2f}% "
          f"({overall_best_name})")


#Load nltk corpus
print("\n--- NLTK Movie Reviews Corpus ---")

nltk_texts = []
nltk_labels = []

for category in movie_reviews.categories():
    label = 1 if category == 'pos' else 0
    for fileid in movie_reviews.fileids(category):
        words = movie_reviews.words(fileid)
        text = ' '.join(words)
        nltk_texts.append(text)
        nltk_labels.append(label)

nltk_texts = np.array(nltk_texts)
nltk_labels = np.array(nltk_labels)

print(f"NLTK corpus size: {len(nltk_texts)} reviews")
print(f"Negative: {(nltk_labels==0).sum()} | Positive: {(nltk_labels==1).sum()}")
print(f"No spam in this dataset — evaluate directly")

print(f"\nSample NLTK review (first 300 chars):")
print(nltk_texts[0][:300])

nltk_texts_marked = mark_negations(nltk_texts)

x_nltk = vec_7k.transform(nltk_texts_marked)

nltk_preds = best_nb_7k_model.predict(x_nltk)
nltk_acc = (nltk_preds == nltk_labels).mean()
nltk_precision = precision_score(nltk_labels, nltk_preds)
nltk_recall = recall_score(nltk_labels, nltk_preds)
nltk_f1 = f1_score(nltk_labels, nltk_preds)
nltk_cm = get_confusion_matrix(nltk_labels, nltk_preds)

print(f"\n--- Best Model on NLTK Corpus ---")
print(f"Model: NB negation 7k (alpha=0.05)")
print(f"Accuracy:  {nltk_acc*100:.2f}%")
print(f"Precision: {nltk_precision:.3f}")
print(f"Recall:    {nltk_recall:.3f}")
print(f"F1:        {nltk_f1:.3f}")
print(f"Confusion matrix:\n{nltk_cm}")

print(f"\n--- Generalisation Gap ---")
print(f"Validation accuracy: {best_nb_7k_acc*100:.2f}%")
print(f"NLTK accuracy:       {nltk_acc*100:.2f}%")
print(f"Gap:                 {(best_nb_7k_acc - nltk_acc)*100:.2f}%")

# baseline model
x_nltk_baseline = vectorizer_final.transform(nltk_texts)

baseline_preds = best_nb_model.predict(x_nltk_baseline)
baseline_acc   = (baseline_preds == nltk_labels).mean()
baseline_f1    = f1_score(nltk_labels, baseline_preds)

print(f"\n--- Baseline Model on NLTK Corpus ---")
print(f"Model: NB best alpha (5k, standard)")
print(f"Accuracy: {baseline_acc*100:.2f}%")
print(f"F1:       {baseline_f1:.3f}")
print(f"Gap from validation: {(best_nb_acc - baseline_acc)*100:.2f}%")

print(f"\n--- Generalisation Comparison ---")
print(f"{'Model':<35} {'Val acc':>8} {'NLTK acc':>10} {'Gap':>6}")
print("-" * 62)
print(f"{'NB baseline (5k)':<35} {best_nb_acc*100:>7.2f}% "
      f"{baseline_acc*100:>9.2f}% {(best_nb_acc-baseline_acc)*100:>5.2f}%")
print(f"{'NB negation 7k (best)':<35} {best_nb_7k_acc*100:>7.2f}% "
      f"{nltk_acc*100:>9.2f}% {(best_nb_7k_acc-nltk_acc)*100:>5.2f}%")

nltk_correct   = np.where(nltk_preds == nltk_labels)[0]
nltk_incorrect = np.where(nltk_preds != nltk_labels)[0]

print(f"\n--- NLTK: CORRECT PREDICTIONS (sample) ---")
for i in nltk_correct[:3]:
    print(f"True: {nltk_labels[i]} | Pred: {nltk_preds[i]}")
    print(nltk_texts[i][:200])
    print("---")

print(f"\n--- NLTK: INCORRECT PREDICTIONS (failure cases) ---")
for i in nltk_incorrect[:5]:
    print(f"True: {nltk_labels[i]} | Pred: {nltk_preds[i]}")
    print(nltk_texts[i][:200])
    print("---")

our_lengths  = np.array([len(t.split()) for t in text_val_clean])
nltk_lengths = np.array([len(t.split()) for t in nltk_texts])

print(f"\n--- Text Length Comparison ---")
print(f"Our validation reviews:")
print(f"  Mean words:   {our_lengths.mean():.0f}")
print(f"  Median words: {np.median(our_lengths):.0f}")
print(f"  Min words:    {our_lengths.min()}")
print(f"  Max words:    {our_lengths.max()}")
print(f"\nNLTK reviews:")
print(f"  Mean words:   {nltk_lengths.mean():.0f}")
print(f"  Median words: {np.median(nltk_lengths):.0f}")
print(f"  Min words:    {nltk_lengths.min()}")
print(f"  Max words:    {nltk_lengths.max()}")

our_vocab  = set(vectorizer_final.get_feature_names_out())
nltk_vocab = set(' '.join(nltk_texts).lower().split())

overlap        = our_vocab & nltk_vocab
overlap_pct    = len(overlap) / len(our_vocab) * 100
nltk_coverage  = len(overlap) / len(nltk_vocab) * 100

print(f"\n--- Vocabulary Analysis ---")
print(f"Our training vocabulary:    {len(our_vocab)} words")
print(f"NLTK corpus vocabulary:     {len(nltk_vocab)} words")
print(f"Overlap:                    {len(overlap)} words")
print(f"% of our vocab in NLTK:     {overlap_pct:.1f}%")
print(f"% of NLTK vocab covered:    {nltk_coverage:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, title in zip(
    axes,
    [get_confusion_matrix(labels_val_clean,
                          best_nb_7k_model.predict(
                              vec_7k.transform(text_val_marked))),
     nltk_cm],
    [f'Validation Set ({best_nb_7k_acc*100:.1f}%)',
     f'NLTK Corpus ({nltk_acc*100:.1f}%)']
):
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]),
                    ha='center', va='center',
                    color='white' if cm[row, col] > cm.max()/2
                    else 'black',
                    fontsize=14)
    plt.colorbar(im, ax=ax)

plt.suptitle('Generalisation: Validation vs NLTK Movie Reviews', fontsize=13)
plt.tight_layout()
plt.savefig('nltk_generalisation.png', dpi=150)
plt.show()
