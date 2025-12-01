# 🕵️‍♂️ Fake Review Detector  
A multi-model system for detecting **computer-generated (CG)** vs **original (OR)** product reviews using NLP, topic modeling, and ensemble learning.

This project explores whether fake reviews can be detected by examining **category relevance**, **rating–text consistency**, and **linguistic style patterns**, and combines these signals into a weighted ensemble for robust classification.

---

## 🚀 Overview

Modern e-commerce platforms face an increasing surge of **AI-generated fake reviews**, which harm product credibility and user trust.  
This project builds a **3-model detection pipeline** that analyzes reviews from multiple perspectives:

### **1. Category–Text Relevance Model (Model 1)**
- Hypothesis: Fake reviews are less semantically aligned with the product category.  
- Uses **LDA (topic modeling)** + **TF-IDF** category vectors.  
- Computes cosine similarity to score relevance.  
- Outputs a probability of a review being Original (OR) or Computer-Generated (CG).

### **2. Rating Consistency Model using BERT (Model 2)**
- Hypothesis: Fake reviews often have text that does not match their given rating.  
- Fine-tunes **BERT** to predict rating from text.  
- Computes the deviation between predicted and actual rating.  
- Larger deviation ⇒ more likely to be CG.

### **3. CBOW + Decision Tree Classifier (Model 3)**
- Hypothesis: CG reviews follow different stylistic patterns.  
- Uses **Bag-of-Words (CBOW embeddings)** + **Decision Tree classifier**.  
- Serves as a direct text-based baseline.

### **🔗 Weighted Ensemble**
Instead of picking one model, this system combines all three using a **data-dependent weighting scheme**:

- Category relevance contributes more for certain categories.  
- Rating model has higher weight for low-rating reviews (1–3).  
- Remaining weight goes to the BoW-Decision Tree model.

This adaptive ensemble achieves better accuracy & generalization.

---

## 📊 Results

### **Validation Set**
| Model | Accuracy |
|------|----------|
| Model 3 only | 72.51% |
| Ensemble (M1 + M2 + M3) | **74.49%** |

### **Test Set**
| Model | Accuracy |
|------|----------|
| Model 3 only | 71.44% |
| Ensemble (M1 + M2 + M3) | **73.11%** |

The ensemble offers a clear improvement over individual models.

---

## 📂 Project Structure

```
Fake_Review_Detector/
├── data/
│   ├── processed/       # Processed data files
│   │   ├── preprocessed_test.csv
│   │   ├── preprocessed_train.csv
│   │   ├── preprocessed_val.csv
│   │   └── val_combined.csv
│   └── raw/             # Raw input data files
│       ├── fake_reviews.csv
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
│
├── notebooks/           # Jupyter notebooks for exploration and visualization
│   └── eda.ipynb
│
├── src/                 # Source code
│   ├── config.py        # Configuration settings
│   ├── embeddings/      # Text vectorization methods
│   │   └── word2vec.py  # Bag of Words vectorization
│   │
│   ├── ensemble/        # Ensemble learning implementation
│   │   ├── __init__.py
│   │   ├── test_probs.csv
│   │   ├── val_probs.csv
│   │   └── weighted_ensemble.py
│   │
│   ├── models/          # ML model implementations
│   │   ├── decision_tree.py
│   │   ├── model1/
│   │   │   ├── category_review_relevance.csv
│   │   │   ├── dataset/
│   │   │   │   ├── test.csv
│   │   │   │   ├── train.csv
│   │   │   │   └── val.csv
│   │   │   ├── inference.py
│   │   │   ├── models/
│   │   │   │   ├── dictionary.gensim
│   │   │   │   ├── label_encoder.pkl
│   │   │   │   ├── lda_model.gensim
│   │   │   │   ├── lda_model.gensim.expElogbeta.npy
│   │   │   │   ├── lda_model.gensim.state
│   │   │   │   └── tfidf_vectorizer.pkl
│   │   │   ├── threshold.py
│   │   │   ├── train.py
│   │   │   ├── val_category_relevance.csv
│   │   │   ├── val_results.csv
│   │   │   └── val_set_metrics.csv
│   │   └── model2/
│   │       ├── dataset.csv
│   │       ├── metrics_with_threshold_per_rating.csv
│   │       ├── model_bert.py
│   │       ├── test.csv
│   │       ├── thresold.py
│   │       ├── train.csv
│   │       └── val.csv
│   │
│   ├── preprocessing/   # Text preprocessing pipeline
│   │   ├── clean_text.py       # Text cleaning functions
│   │   ├── lemmatization.py    # Word lemmatization
│   │   ├── preprocessing_pipeline.py  # Complete pipeline
│   │   └── stemming.py         # Word stemming
│   │
│   ├── main.py          # Main script to run the project
│   └── utils.py         # Utility functions
│
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── Report.pdf           # Project report
```
