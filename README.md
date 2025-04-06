# 🧠 Bayesian Inference & Naive Bayes Classification

This project showcases my implementation of:

1. **Variable Elimination in Bayesian Networks** (Q1)
2. **Naive Bayes Classifier for Diabetes Prediction** (Q2)

Built as part of my Artificial Intelligence coursework at Penn State, this project gave me hands-on experience with probabilistic reasoning, inference under uncertainty, and learning from data.

---

## 📂 Project Files

| File Name | Description |

| `solution_q1.py` | Implements **variable elimination** on a 5-node Bayesian Network to compute conditional distributions like P(Burglary | JohnCalls = +j) |

| `solution_q2.py` | Implements a **Naive Bayes classifier** using real patient data to predict diabetes from glucose and blood pressure levels |

---

## 🔍 Q1: Bayesian Network Inference with Variable Elimination

### 🔧 Features:

-   Implements variable elimination with factor operations: `flattening`, `product`, `marginalization`, `evidence application`, and `normalization`
-   Supports evidence-based inference like:  
    `P(Burglary | JohnCalls = +j)`  
    `P(Burglary | MaryCalls = +m)`  
    `P(Burglary | JohnCalls = +j, MaryCalls = +m)`
-   Prints the full probability distribution table for the query variable after eliminating hidden variables

---

## 🔬 Q2: Naive Bayes Prediction for Diabetes Diagnosis

### 📊 Dataset:

Uses a dataset of 995 records with:

-   `glucose` level (X1)
-   `blood pressure` level (X2)
-   `diabetes` diagnosis (Y ∈ {0, 1})

### 🧩 What the Model Does:

-   Computes:
    -   `P(Y)` — prior
    -   `P(X1 | Y)` — glucose likelihood
    -   `P(X2 | Y)` — BP likelihood
-   Generates a lookup table for `P(Y | X1, X2)` using Naive Bayes assumptions
-   Predicts diabetes for new patients and evaluates accuracy

### 📋 Sample Output Includes:

-   Part-wise printed outputs (e.g., 2.1.1, 2.2.1, 2.3.2)
-   Probabilities and lookup tables
-   Accuracy of the model on the full dataset
-   Example prediction:  
    `Sample instance {'glucose': 50, 'bloodpressure': 75} → Predicted diabetes: 1`

### 🎯 Accuracy:

Achieves strong predictive performance using basic probabilistic modeling — with no complex tuning.

---

## 🧠 What I Learned

-   Implementing **exact inference** using factor operations gave me a solid foundation in probabilistic reasoning
-   Coding a **Naive Bayes classifier** from scratch helped me understand how real-world data maps to theoretical probability models
-   Learned to debug probabilistic code and validate correctness using expected values

---

## 📌 Possible Improvements

-   Add user input to query arbitrary variables and evidence combinations (Q1)
-   Extend Q2 to handle unseen feature values using Laplace smoothing
-   Visualize the Bayesian network structure and factor graph operations

---

### ▶️ How to Run:

    python3 solution_q1.py
    or
    python3 solution_q2.py

---

---

## 🧠 Let’s Connect!

**Tej Jaideep Patel**  
B.S. Computer Engineering  
📍 Penn State University  
✉️ tejpatelce@gmail.com  
📞 814-826-5544

---
