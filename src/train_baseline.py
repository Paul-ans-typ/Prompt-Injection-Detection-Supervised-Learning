import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    train_path = 'data/processed/train.csv'
    test_path = 'data/processed/test.csv'
    model_dir = 'models/'

    os.makedirs(model_dir, exist_ok=True)

    print("Loading processed data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Ensure strings and handle any stray NaNs
    train_df = train_df.dropna(subset=['Prompt', 'isMalicious'])
    test_df = test_df.dropna(subset=['Prompt', 'isMalicious'])
    
    X_train = train_df['Prompt'].astype(str)
    y_train = train_df['isMalicious']
    X_test = test_df['Prompt'].astype(str)
    y_test = test_df['isMalicious']

    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000) 
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating model on test set...")
    y_pred = model.predict(X_test_tfidf)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"Saving model and vectorizer to {model_dir}...")
    joblib.dump(model, os.path.join(model_dir, 'baseline_lr_model.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
    print("Done!")

if __name__ == "__main__":
    main()