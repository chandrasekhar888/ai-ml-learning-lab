# AI & ML Learning Lab

<img src="assets/ai-lab-banner.svg" alt="AI & ML Learning Lab banner" width="100%" />

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-D00000?logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Coming%20Soon-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-2C7A7B)
![Streamlit](https://img.shields.io/badge/Streamlit-Apps-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

This repository is a personal AI and machine learning learning lab where I collect current experiments, coursework, and future projects. It serves as a portfolio and a place to document what I build, what I learn, and what I plan to explore next.

## Featured projects

| Project | Focus | Status | Links |
| --- | --- | --- | --- |
| Customer Churn Prediction — ANN | Artificial neural network for churn classification | Deployed project | [Live demo](https://customer-churn-ann-ap.streamlit.app/) • [Repository](https://github.com/chandrasekhar888/Customer_Churn_ANN) |
| IMDb Sentiment Analysis — RNN | Recurrent neural network for text sentiment | Deployed project | [Live demo](https://imdb-sentiment-rnn-ap.streamlit.app/) • [Repository](https://github.com/chandrasekhar888/Customer_Churn_ANN) |
| Hamlet Next-Word Prediction — LSTM | LSTM-based next-word prediction using Shakespeare's Hamlet | Local learning project in progress | [LSTM folder](LSTM/) • dataset from NLTK Gutenberg |

## Why this repository exists

This lab brings together:

- model experiments and notebooks
- data preprocessing workflows
- deployment-ready portfolio apps
- reusable learning notes and project documentation
- future work across NLP, deep learning, classical ML, and MLOps

It is intended to grow over time as a central place for AI and ML study, experimentation, and portfolio-building.

## Future learning roadmap

1. Strengthen foundational ML and deep learning theory.
2. Expand NLP projects with transformers and attention-based models.
3. Add model evaluation, explainability, and deployment workflows.
4. Build end-to-end ML projects with reproducible pipelines and documentation.
5. Continue exploring production-ready AI applications with Streamlit, APIs, and cloud deployment.

## Repository structure

```text
AI-ML-Learning-Lab/
├── LSTM/
│   ├── app.py                    # Streamlit app entry point
│   ├── data_collection_preprocessing.ipynb
│   ├── next_word_gru.ipynb       # GRU notebook for future work
│   ├── next_word_lstm.ipynb      # LSTM training and prediction notebook
│   ├── README.md                 # LSTM project notes
│   ├── requirements.txt          # Python dependencies
│   └── .venv/                    # local environment only; typically ignored
├── assets/
│   └── ai-lab-banner.svg         # animated repository banner
├── .gitignore
├── README.md
└── LICENSE (if added later)
```

## Installation and reproducibility notes

For the local LSTM project:

```bash
cd D:\AI_Projects_Practice
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r LSTM\requirements.txt
```

Then run the notebook workflow in the `LSTM` folder:

```python
import nltk
nltk.download('gutenberg')
```

The project uses:

- NLTK Gutenberg for the Shakespeare corpus
- TensorFlow / Keras for the LSTM model
- NumPy and scikit-learn for preprocessing and data splitting
- Jupyter notebooks for data preparation and training

## Current project status

- Customer Churn Prediction — ANN: linked externally and available through the deployed Streamlit app.
- IMDb Sentiment Analysis — RNN: linked externally and available through the deployed Streamlit app.
- Hamlet Next-Word Prediction — LSTM: preprocessing, sequence generation, padding, model training, prediction logic, and artifact saving are implemented in the notebooks and files in the `LSTM` folder. The code shows overfitting during training, and further evaluation and deployment work may remain.
- GRU notebook: present as a planned follow-up, not a completed implementation in the current code.

## Notes on dataset and generated artifacts

- Local training artifacts such as `.venv`, model files (`.h5`, `.keras`), tokenizer dumps (`.pickle`), and generated data files remain out of Git so the portfolio stays focused on source code, notebooks, and documentation.
- The LSTM notebooks can regenerate the corpus and preprocessing outputs locally when needed, without storing those generated artifacts in the repository history.

## Contact and portfolio intent

This repository is designed as a portfolio and learning archive for applied AI and ML work. It is intentionally structured so that new projects can be added over time without disrupting the existing learning materials.
