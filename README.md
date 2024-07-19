
# Event-Driven-Stock-Prediction-Using-NLP

This project leverages Natural Language Processing (NLP) techniques to perform sentiment analysis on Reuters News headlines to predict stock prices. The core of the project involves connecting Bayesian Deep Neural Networks (DNN) with stock price prediction.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The objective of this project is to develop a system that can predict stock prices based on sentiment analysis of news headlines. By integrating Bayesian DNNs, the project aims to enhance the accuracy of predictions. The workflow involves several key steps including data collection, preprocessing, feature engineering, training the model, and making predictions.

## Features

- **Sentiment Analysis:** Extract sentiments from Reuters News headlines.
- **Bayesian DNN:** Use Bayesian Deep Neural Networks for stock price prediction.
- **Data Collection:** Collect and preprocess data for training and prediction.
- **Feature Engineering:** Generate features from news headlines and historical stock prices.
- **Prediction:** Predict stock prices based on the sentiment of news headlines.

## Technologies

- **Python:** A powerful programming language for data analysis and machine learning.
- **Pandas:** A data manipulation and analysis library for Python.
- **NumPy:** A library for numerical computations in Python.
- **Scikit-learn:** A machine learning library for Python.
- **TensorFlow:** An open-source platform for machine learning.
- **Keras:** A high-level neural networks API.
- **NLTK (Natural Language Toolkit):** A library for working with human language data.
- **Reuters News Dataset:** A dataset containing news headlines from Reuters.
- **Bayesian DNN:** A type of neural network that incorporates Bayesian inference.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/stock-prediction-nlp.git
    cd stock-prediction-nlp
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Collect and preprocess the data:**
    ```sh
    python src/data_preprocessing.py
    ```

2. **Perform feature engineering:**
    ```sh
    python src/feature_engineering.py
    ```

3. **Train the Bayesian DNN model:**
    ```sh
    python src/train_model.py
    ```

4. **Make predictions:**
    ```sh
    python src/predict.py
    ```

## Project Structure

```plaintext
.
├── data
│   ├── raw
│   ├── processed
├── notebooks
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Results

The project successfully integrates sentiment analysis with stock price prediction. The Bayesian DNN model shows improved accuracy in predictions, providing valuable insights for potential investors.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the repository:** Click the "Fork" button on the upper right corner of the repository page.

2. **Create a new branch:**
    ```sh
    git checkout -b feature-branch
    ```

3. **Make your changes and commit them:**
    ```sh
    git commit -m 'Add new feature'
    ```

4. **Push to the branch:**
    ```sh
    git push origin feature-branch
    ```

5. **Create a new Pull Request:** Go to the repository on GitHub and click the "New pull request" button.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

