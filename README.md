# Instagram-data-analysis-
This project analyzes Instagram posts and predicts the sentiment of the captions using a machine learning model.

# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Instagram dataset
def load_instagram_data():
    # Synthetic Instagram dataset
    data = {
        'Caption': [
            "Loving the vibes at the beach! ",
            "Had a bad day, but trying to stay positive. ",
            "Best coffee ever at this cute little cafe!",
            "Feeling so overwhelmed with work.",
            "What a fantastic hike with the best view!"
        ],
        'Sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    }
    return pd.DataFrame(data)

# Step 2: Preprocess the data
def preprocess_instagram_data(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Caption'])
    y = df['Sentiment']
    return X, y, vectorizer

# Step 3: Train a sentiment analysis model
def train_sentiment_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Step 4: Evaluate the sentiment model
def evaluate_sentiment_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# Main function
def main():
    # Load and preprocess data
    data = load_instagram_data()
    X, y, vectorizer = preprocess_instagram_data(data)

    # Train the sentiment analysis model
    model, X_test, y_test = train_sentiment_model(X, y)

    # Evaluate the model
    accuracy, report = evaluate_sentiment_model(model, X_test, y_test)

    # Print results
    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    # Test on new captions
    new_captions = ["What a gorgeous sunset!", "Feeling a bit down today."]
    new_X = vectorizer.transform(new_captions)
    new_predictions = model.predict(new_X)

    for caption, sentiment in zip(new_captions, new_predictions):
        print(f"Caption: {caption} | Predicted Sentiment: {sentiment}")

# Run the project
if __name__ == "__main__":
    main()
