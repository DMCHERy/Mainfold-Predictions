

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import openai
import os
from dotenv import load_dotenv

# Load your OpenAI API key from the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def fetch_markets():
    url = "https://api.manifold.markets/v0/markets"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def filter_binary_resolved(markets):
    return [
        m for m in markets
        if m['outcomeType'] == 'BINARY'
        and m.get('isResolved') is True
        and m.get('resolution') in ['YES', 'NO']
    ]

def build_feature_set(markets):
    data = []
    for m in markets:
        try:
            gpt_analysis = get_gpt_opinion_summary(m['question'])
            feature = {
                'id': m['id'],
                'question': m['question'],
                'questionLength': len(m['question']),
                'volume': m.get('volume', 0),
                'numTraders': m.get('uniqueBettorCount', 0),
                'timeOpen': (m['closeTime'] - m['createdTime']) / (1000 * 60 * 60 * 24),
                'gpt_pos_confidence': gpt_analysis.get('yes_confidence', 0.5),
                'label': 1 if m['resolution'] == 'YES' else 0
            }
            data.append(feature)
        except KeyError:
            continue
    return pd.DataFrame(data)
    

def get_gpt_opinion_summary(question):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful analyst. For a given prediction market question, estimate the likelihood of a YES outcome."},
            {"role": "user", "content": f"Question: {question}\nWhat is the probability (0 to 1) that the answer will be YES? Just return the number."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5
        )
        reply = response.choices[0].message.content.strip()
        score = float(reply) if 0 <= float(reply) <= 1 else 0.5
        return {'yes_confidence': score}
    except:
        return {'yes_confidence': 0.5}

def train_and_evaluate(df):
    X = df.drop(columns=["id", "question", "label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Model Accuracy: {acc * 100:.2f}%")

    print("\n  Importance Scale")
    feat_scores = sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])
    for feat, score in feat_scores:
        print(f"{feat}: {score:.4f}")


    plt.figure(figsize=(4, 4))
    plt.bar([f[0] for f in feat_scores], [f[1] for f in feat_scores])
    plt.xticks(rotation=45)
    plt.title("How Important is this Data")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    print("\n How This Works:")
    print("""This model learns from past Manifold markets to predict their market outcomes.
Features include text length, trading volume, time open, and a GPT.
The model is a Random Forest classifier trained on resolved binary markets
-there is nothing special just simple annylysis in this model-
it wont have the best reasults since it only uses recent data from Mainfolds Markets.
""")
    return clf


def fetch_unresolved_binary_markets():
    url = "https://api.manifold.markets/v0/markets"
    response = requests.get(url)
    response.raise_for_status()
    markets = response.json()
    return [
        m for m in markets
        if m['outcomeType'] == 'BINARY'
        and not m.get('isResolved', False)
    ]

def build_features_for_unresolved(markets):
    data = []
    ids = []
    questions = []

    for m in markets:
        try:
            gpt_score = get_gpt_opinion_summary(m['question'])
            features = {
                'questionLength': len(m['question']),
                'volume': m.get('volume', 0),
                'numTraders': m.get('uniqueBettorCount', 0),
                'timeOpen': (m['closeTime'] - m['createdTime']) / (1000 * 60 * 60 * 24),
                'gpt_pos_confidence': gpt_score.get('yes_confidence', 0.5),
            }
            data.append(features)
            ids.append(m['id'])
            questions.append(m['question'])
        except:
            continue

    df = pd.DataFrame(data)
    df['id'] = ids
    df['question'] = questions
    return df

def predict_unresolved_markets(model):
    print("\nFetching unresolved markets...")
    unresolved = fetch_unresolved_binary_markets()
    if not unresolved:
        print("No unresolved binary markets found.")
        return

    df_features = build_features_for_unresolved(unresolved)
    X_unresolved = df_features.drop(columns=["id", "question"])

    print("Making predictions...")
    probs = model.predict_proba(X_unresolved)[:, 1]
    df_features['predicted_prob_yes'] = probs

    df_sorted = df_features.sort_values(by='predicted_prob_yes', ascending=False)

    print("\nTop Predictions:")
    print(df_sorted[['question', 'predicted_prob_yes']].head(10))

    df_sorted.to_csv("manifold_predictions.csv", index=False)
    print("Saved as 'manifold_predictions.csv'")

if __name__ == "__main__":
    try:
        print("Fetching market data from Manifold...")
        markets = fetch_markets()

        print("Filtering binary resolved markets...")
        binary_markets = filter_binary_resolved(markets)

        print("Building feature set...")
        df = build_feature_set(binary_markets)

        if df.empty:
            print("No binary resolved markets found.")
        else:
            print(f"Loaded {len(df)} valid markets.")
            clf = train_and_evaluate(df)
            predict_unresolved_markets(clf)

    except Exception as e:
        print(f"Error occurred: {e}")

