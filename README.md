![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-Alpha-red)
#  Predicting Manifold Markets Using GPT + Random Forest

This project uses real-time data from [Manifold Markets](https://manifold.markets) to predict outcomes of unresolved **binary prediction markets**.

It combines:
-  Market data scraped from the public API
-  A Random Forest model trained on past resolved markets
-  GPT-3.5 as a probabilistic reasoning signal



##  How It Works?

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


## GPT Integration
Each market question gets sent to GPT-3.5 and it tries to make a smart guess based on how the question is worded. I ask it to treat every market like a yes/no problem and do its best.
 ```
Example prompt: “Please try to look at this question as a binary problem and predict this to the best of your ability.”
 ```
It’s kind of like giving the model a voice in the room — a way to simulate outside opinions or public vibes.

##  Model Performance Testing

You will need to run the [Mainfold Predictor](https://github.com/DMCHERy/Mainfold-Predictions/blob/main/Mainfold%20Predictor.py)
Only then you will be able to check your prediction using the [Mainfold_Backtester](https://github.com/DMCHERy/Mainfold-Predictions/blob/main/Markets_Backtester.py)  The Manifold Predictor doesn't use the results of resolved markets — it's only showcasing the predictions, without having to wait months or years for those markets to resolve. Even though it's just for fun, it does predict future markets — and technically (not saying you should, for legal reasons), it could give you an edge when guessing outcomes.


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

4. **If you have problems**
   
   I've only ran into one problem so far
   
    if you have an error in the heading part of the code try running
   ```
   ! pip install python-dotenv
   ```



##  Limitations

-Quite small scale 

-Needs Better Understandfing of the Mainfold Markets with the APIs of News corp's

-Many Many more data bases needed for it to be extremely accurate


##  License

This project is licensed under the **MIT License**.



##  Tags

`#machine-learning` `#gpt3` `#openai` `#randomforest` `#manifoldmarkets` `#ai-prediction` `#college-project` `#python`




