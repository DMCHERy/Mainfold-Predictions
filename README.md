#  Predicting Manifold Markets Using GPT + Random Forest

This project uses real-time data from [Manifold Markets](https://manifold.markets) to predict outcomes of unresolved **binary prediction markets**.

It combines:
-  Market data scraped from the public API
-  A Random Forest model trained on past resolved markets
-  GPT-3.5 as a probabilistic reasoning signal



##  How It Works

### 1. Market Data Collection
- Retrieves market data from Manifold's public API.
- Filters for **binary** and **resolved** markets.

### 2. Feature Engineering
Extracts structured features such as:
- `questionLength` – length of the market question
- `volume` – amount traded
- `numTraders` – unique participants
- `timeOpen` – how long the market was open
- `gpt_pos_confidence` – GPT-estimated probability the answer is YES

### 3. Modeling
- A `RandomForestClassifier` is trained on resolved markets.

### 4. Prediction
- Unresolved markets are scored based on model predictions.
- A ranked list is saved to `manifold_predictions.csv`.



##  GPT Integration
All market questions are passed to OpenAI GPT-3.5 with the following prompt:

```
You are a helpful analyst. For a given prediction market question, estimate the probability (0 to 1) that the answer will be YES.

Q: Will Anthropic release Claude 3.5 before August 2024?
A: 0.72
```

Used as a **proxy for public sentiment** or outside priors.



##  Model Performance

- **Accuracy**: ~43% (on small recent sample)
- **Feature Importances:**
  - `timeOpen`: 0.32
  - `questionLength`: 0.25
  - `volume`: 0.23
  - `numTraders`: 0.20
  - `gpt_confidence`: ~0.15



##  Files Included

| File | Description |
|------|-------------|
| `manifold_model.py` | Main codebase for training and prediction |
| `.env` | Holds your OpenAI API key (not included) |
| `manifold_predictions.csv` | Model predictions on unresolved markets |
| `feature_importance.png` | Bar chart of model feature weights |



##  Setup & Installation

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/market-predictor.git
cd market-predictor
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Set up your API key**
Create a `.env` file with the line:
```
OPENAI_API_KEY=your-openai-api-key-here
```
Make sure `.env` is included in `.gitignore`!



##  Example Output

```
Top Predictions:
Q: Will "Surprising LLM reasoning failures" be in the mainstream media? → 0.78
Q: Will a major U.S. bank fail by the end of the year? → 0.61
Q: Will Trump return to Twitter before November 2024? → 0.59
```



##  Limitations

- Small sample size may limit generalization
- GPT answers are not live probability trackers
- Does not yet analyze comment sentiment or trader history


##  License

This project is licensed under the **MIT License**.



##  Tags

`#machine-learning` `#gpt3` `#openai` `#randomforest` `#manifoldmarkets` `#ai-prediction` `#college-project` `#python`




