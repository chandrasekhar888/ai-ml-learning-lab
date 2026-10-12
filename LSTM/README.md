# Next Word Prediction with LSTM

This project focuses on a next-word prediction task using Shakespeare's *Hamlet* dataset. The notebook workflow builds the preprocessing pipeline, generates n-gram sequences, pads them for training, and trains an LSTM model to predict the next token in a sentence.

## Workflow in the notebooks

### 1. Data collection and preprocessing

The notebook `data_collection_preprocessing.ipynb` downloads the Gutenberg corpus and saves the raw Hamlet text to `hamlet.txt`.

```python
import nltk
nltk.download('gutenberg')

from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-hamlet.txt')

with open('hamlet.txt', 'w', encoding='utf-8') as file:
    file.write(data)
```

The text is then read and normalized to lowercase before tokenization:

```python
with open('hamlet.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()
```

### 2. Tokenization and vocabulary size

The text is tokenized with Keras `Tokenizer`, which builds a vocabulary from the corpus and assigns integer IDs to each word.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
```

This creates the vocabulary size used later in the embedding layer and output softmax layer.

### 3. N-gram sequence generation

The notebooks generate training examples from overlapping word sequences. For each line, the code builds all possible next-word windows from the token list.

```python
input_sequences = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
```

This is a standard language-modeling approach: given the first $n-1$ words, the model learns to predict the $n$th word.

### 4. Padding and sequence shaping

The notebook normalizes the n-grams to a common length using `pad_sequences` so each sequence can be fed into a recurrent model.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
)
```

The data is then split into input and target arrays:

```python
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
```

This means each input sequence is the prefix of a phrase, and the target is the final word to predict.

### 5. Train/test split and preprocessing artifact

The project uses `train_test_split` to separate training and validation/test data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

The preprocessed arrays are saved with `pickle` so the dataset can be reused without rerunning the whole tokenization pipeline:

```python
import pickle

with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "total_words": total_words,
        "max_sequence_len": max_sequence_len,
        "tokenizer": tokenizer
    }, f)
```

The notebook also contains an exploratory one-hot-label step for softmax classification:

```python
import tensorflow as tf

y = tf.keras.utils.to_categorical(y, num_classes=total_words)
```

The file comments note that integer targets are often kept as-is to avoid creating a very large one-hot matrix when the vocabulary is large. In other words, the model can train with sparse categorical cross-entropy instead of a full one-hot target representation.

## Model architecture

The LSTM notebook defines a sequential model with an embedding layer followed by stacked LSTM layers and a final dense classification layer:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input

model = Sequential([
    Input(shape=(max_sequence_len - 1,)),
    Embedding(input_dim=total_words, output_dim=100),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation="softmax")
])
```

The model is compiled with categorical cross-entropy and accuracy as the metric:

```python
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

The training step uses validation data:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1
)
```

## Prediction function

The notebook includes a prediction helper that turns a text input into a token sequence, pads it, and selects the most likely next word:

```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predicted_probabilities = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probabilities, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return None
```

This function is then used on a phrase such as:

```python
input_text = "to be or not to be"
next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
print(next_word)
```

## Artifact saving and reproducibility

The relevant artifacts created in this project are:

- `hamlet.txt` — downloaded corpus text
- `preprocessed_data.pkl` — saved training/test arrays, vocabulary size, and tokenizer
- `next_word_lstm.h5` — saved Keras model
- `next_word_lstm.keras` — alternate Keras model save
- `tokenizer.pickle` — pickled tokenizer object

These files are useful for local experimentation and model reuse, but they are generated outputs rather than the source code itself.

## Observed training behavior and overfitting

The notebook output shows the pattern of overfitting clearly:

- training accuracy rises across epochs
- validation accuracy stays very low
- validation loss increases as training continues

The recorded outputs show a pattern like:

- training accuracy rising from roughly 0.17 to 0.24
- validation accuracy staying around 0.056
- validation loss increasing from about 8.46 to about 9.11

This indicates that the model fits the training set but fails to generalize to unseen validation data. In other words, the network is memorizing patterns from the training sequences rather than learning a robust predictive distribution for next words.

This is an important learning signal for the project: the LSTM training is implemented and runs, but the model is not yet considered a production-ready or strongly generalized text predictor. Further evaluation, regularization, tuning, and validation work would be needed before it is ready for broader use or deployment.

## GRU track

The notebook `next_word_gru.ipynb` is a starter file and is not a completed GRU implementation yet. It is marked as a planned follow-up after the LSTM stage.

## Project status

Status as confirmed by the current source code:

- dataset acquisition from NLTK Gutenberg: implemented
- text preprocessing and tokenization: implemented
- n-gram sequence generation and padding: implemented
- LSTM training and next-word prediction: implemented
- validation overfitting observed: confirmed in notebook output
- full deployment, robust evaluation, and GRU completion: not yet completed in the codebase
