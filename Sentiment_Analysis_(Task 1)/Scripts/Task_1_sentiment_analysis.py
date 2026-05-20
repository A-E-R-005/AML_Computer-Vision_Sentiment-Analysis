import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler

from task_1 import vectorizer
from task_1_filter import x_train_final, labels_train_clean, x_val_final, get_confusion_matrix, labels_val_clean, \
    vectorizer_final, text_val_clean, text_train_clean, text_test_clean

# Logistic Regression (BaseLine)

lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0
)

lr.fit(x_train_final, labels_train_clean)

lr_val_preds = lr.predict(x_val_final)
lr_val_probs = lr.predict_proba(x_val_final)[:, 1]

lr_cm = get_confusion_matrix(labels_val_clean, lr_val_preds)
lr_acc = (lr_val_preds == labels_val_clean).mean()
lr_precision = precision_score(labels_val_clean, lr_val_preds)
lr_recall = recall_score(labels_val_clean, lr_val_preds)
lr_f1 = f1_score(labels_val_clean, lr_val_preds)

print(f"\n--- Logistic Regression (Baseline) ---")
print(f"Accuracy:  {lr_acc*100:.2f}%")
print(f"Precision: {lr_precision:.3f}")
print(f"Recall:    {lr_recall:.3f}")
print(f"F1:        {lr_f1:.3f}")
print(f"Confusion matrix:\n{lr_cm}")

# Most positive and negative words according to model weights
feature_names_final = vectorizer_final.get_feature_names_out()
lr_coeffs = lr.coef_[0]

top_positive_idx = lr_coeffs.argsort()[-15:][::-1]
top_negative_idx = lr_coeffs.argsort()[:15]

print(f"\nTop positive words (predicts label=1):")
for idx in top_positive_idx:
    print(f"  '{feature_names_final[idx]}': {lr_coeffs[idx]:.3f}")

print(f"\nTop negative words (predicts label=0):")
for idx in top_negative_idx:
    print(f"  '{feature_names_final[idx]}': {lr_coeffs[idx]:.3f}")

# Correct and incorrect predictions
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

# Naive Bayes
nb = MultinomialNB(alpha=1.0)
nb.fit(x_train_final, labels_train_clean)

nb_val_preds = nb.predict(x_val_final)
nb_cm = get_confusion_matrix(labels_val_clean, nb_val_preds)
nb_acc = (nb_val_preds == labels_val_clean).mean()
nb_precision = precision_score(labels_val_clean, nb_val_preds)
nb_recall = recall_score(labels_val_clean, nb_val_preds)
nb_f1 = f1_score(labels_val_clean, nb_val_preds)

print(f"\n--- Naive Bayes ---")
print(f"Accuracy:  {nb_acc*100:.2f}%")
print(f"Precision: {nb_precision:.3f}")
print(f"Recall:    {nb_recall:.3f}")
print(f"F1:        {nb_f1:.3f}")
print(f"Confusion matrix:\n{nb_cm}")

# Linear SVM
svm_clf = LinearSVC(max_iter=2000, random_state=42, C=1.0)
svm_clf.fit(x_train_final, labels_train_clean)

svm_val_preds = svm_clf.predict(x_val_final)
svm_cm = get_confusion_matrix(labels_val_clean, svm_val_preds)
svm_acc = (svm_val_preds == labels_val_clean).mean()
svm_precision = precision_score(labels_val_clean, svm_val_preds)
svm_recall = recall_score(labels_val_clean, svm_val_preds)
svm_f1 = f1_score(labels_val_clean, svm_val_preds)

print(f"\n--- Linear SVM ---")
print(f"Accuracy:  {svm_acc*100:.2f}%")
print(f"Precision: {svm_precision:.3f}")
print(f"Recall:    {svm_recall:.3f}")
print(f"F1:        {svm_f1:.3f}")
print(f"Confusion matrix:\n{svm_cm}")


# Logistic Regression C Sweep
print(f"\n--- Logistic Regression C sweep ---")
print(f"{'C':<10} {'Accuracy':>10} {'F1':>8}")
print("-" * 30)

best_lr_acc = 0
best_lr_c = 1.0
best_lr_model = None

for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    lr_c = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr_c.fit(x_train_final, labels_train_clean)
    preds = lr_c.predict(x_val_final)
    acc = (preds == labels_val_clean).mean()
    f = f1_score(labels_val_clean, preds)
    print(f"C={C:<8} {acc*100:>10.2f}% {f:>8.3f}")
    if acc > best_lr_acc:
        best_lr_acc   = acc
        best_lr_c     = C
        best_lr_model = lr_c

print(f"\nBest C: {best_lr_c} → accuracy {best_lr_acc*100:.2f}%")


# Naive Bayes Alpha Sweep
print(f"\n--- Naive Bayes alpha sweep ---")
print(f"{'Alpha':<10} {'Accuracy':>10} {'F1':>8}")
print("-" * 30)

best_nb_acc = 0
best_nb_alpha = 1.0
best_nb_model = None

for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    nb_a = MultinomialNB(alpha=alpha)
    nb_a.fit(x_train_final, labels_train_clean)
    preds = nb_a.predict(x_val_final)
    acc = (preds == labels_val_clean).mean()
    f = f1_score(labels_val_clean, preds)
    print(f"alpha={alpha:<5} {acc * 100:>10.2f}% {f:>8.3f}")
    if acc > best_nb_acc:
        best_nb_acc = acc
        best_nb_alpha = alpha
        best_nb_model = nb_a

print(f"\nBest alpha: {best_nb_alpha} → accuracy {best_nb_acc * 100:.2f}%")

# Model comparison
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

# Confusion matrix visualisation
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

plt.suptitle('Confusion Matrices: Validation Set', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()


#Bigrams
vectorizer_bigram = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    min_df=2,
    sublinear_tf=True,
    ngram_range=(1, 2)
)
vectorizer_bigram.fit(text_train_clean)
x_train_bigram = vectorizer_bigram.transform(text_train_clean)
x_val_bigram = vectorizer_bigram.transform(text_val_clean)
x_test_bigram = vectorizer_bigram.transform(text_test_clean)

print(f"Bigram TF-IDF shape: {x_train_bigram.shape}")

# Naive Bayes and Logistic Regression with Bigram
nb_bigram = MultinomialNB(alpha=0.01)
nb_bigram.fit(x_train_bigram, labels_train_clean)
nb_bigram_preds = nb_bigram.predict(x_val_bigram)
nb_bigram_acc   = (nb_bigram_preds == labels_val_clean).mean()
nb_bigram_f1    = f1_score(labels_val_clean, nb_bigram_preds)
print(f"NB bigram: {nb_bigram_acc*100:.2f}% | F1: {nb_bigram_f1:.3f}")

lr_bigram = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr_bigram.fit(x_train_bigram, labels_train_clean)
lr_bigram_preds = lr_bigram.predict(x_val_bigram)
lr_bigram_acc   = (lr_bigram_preds == labels_val_clean).mean()
lr_bigram_f1    = f1_score(labels_val_clean, lr_bigram_preds)
print(f"LR bigram: {lr_bigram_acc*100:.2f}% | F1: {lr_bigram_f1:.3f}")

