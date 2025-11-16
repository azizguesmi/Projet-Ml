# ----------------------------
# Imports
import pandas as pd
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import nltk
import matplotlib.pyplot as plt

# ----------------------------
# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')

# ----------------------------
# Preprocessing Setup
stop_words = set(stopwords.words('english'))

def preprocess_sarcasm_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Sarcasm markers
    has_caps = bool(re.search(r'[A-Z]{2,}', text))
    has_exclaim = bool(re.search(r'!{2,}', text))
    has_question = bool(re.search(r'\?{2,}', text))
    
    # Emojis to words
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 1 and w not in stop_words]
    
    result = " ".join(tokens)
    
    # Add sarcasm markers
    if has_caps: result += " ALLCAPS"
    if has_exclaim: result += " MULTIEXCLAIM"
    if has_question: result += " MULTIQUESTION"
    
    return result

# ----------------------------
# Load Dataset
df = pd.read_csv("train.En.csv")  # Must have 'tweet' and 'sarcastic'
df['clean_text'] = df['tweet'].apply(preprocess_sarcasm_text)

X = df['clean_text']
y = df['sarcastic']

# ----------------------------
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Handle Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

# ----------------------------
# Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "Naive Bayes": MultinomialNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ----------------------------
# Train, Evaluate & Display Confusion Matrices
best_model_name = None
best_model = None
best_f1 = 0
results = {}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_vec)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-score": f1,
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n=== {name} ===")
    print("Accuracy:", results[name]["Accuracy"])
    print("F1-score:", results[name]["F1-score"])
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Visual confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=results[name]["Confusion Matrix"], display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.show()
    
    # Track best model by F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

# ----------------------------
print(f"\nBest Model: {best_model_name} with F1-score: {best_f1:.4f}")
