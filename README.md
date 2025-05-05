# AI-Powered Sentiment Analysis and Topic Modeling System

This project provides an end-to-end pipeline for performing sentiment analysis and topic modeling on textual data. It uses advanced NLP techniques and Machine Learning models to extract insights from text.

## Features
- Sentiment classification using pre-trained transformer models.
- Topic modeling to identify key discussion points.
- Interactive dashboard for data visualization.

## Getting Started

### Prerequisites
- Python 3.8 or above.
- Install dependencies: `pip install -r requirements.txt`.

### Project Structure
- `data/`: Raw and processed data.
- `notebooks/`: Jupyter notebooks for EDA and modeling.
- `src/`: Source code for the pipeline, models, and dashboard.
- `tests/`: Unit tests for the models.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/mohd-kaif-11/ai-sentiment-topic-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-sentiment-topic-analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the sentiment model:
   ```bash
   python src/sentiment_model.py
   ```
5. Extract topics:
   ```bash
   python src/topic_model.py
   ```
6. Run the dashboard:
   ```bash
   streamlit run src/dashboard.py
   ```

## License
This project is licensed under the MIT License.
