# 🚨 ProjectDeepNLP — Team Timidez (Group A)

Welcome to **ProjectDeepNLP**, a system for automatic alert generation using **Named Entity Recognition (NER)** and **Sentiment Analysis (SA)**. This project was developed by **Team Timidez** from **Group A** as part of the DL+NLP Final Project (2024/2025) at **Universidad Pontificia Comillas (ICAI)**.

We implemented the **intermediate version (up to 9.0 points)** with a joint model for NER and SA and a neural component for generating context-specific alerts.

---

## 🧠 What does it do?

The system processes news articles and social media content to:

- 🏷️ Identify entities (NER)
- 😊 Determine sentiment (SA)
- 📢 Generate alerts (e.g., “Reputation risk: [Entity]”)

---

## ⚙️ How to run the project

Follow these steps in order to reproduce our results:

1. **Download and prepare datasets**
   - Run:
     ```
     data/load_data/load_data.ipynb
     ```

2. **Load Sentiment Analysis model**
   - Run:
     ```
     data/load_data/load_SA_model.ipynb
     ```

3. **Combine NER + SA datasets**
   - Run:
     ```
     data/load_data/combine_NER_SA.ipynb
     ```

4. **Download Word2Vec embeddings**
   - Download `GoogleNews-vectors-negative300.bin.gz` from:
     [GoogleNews Download](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)
   - Place the file inside the `models/` folder.

5. **Train the joint model**
   - Run:
     ```
     python src/train.py
     ```

6. **Evaluate the model and generate alerts**
   - Run:
     ```
     python src/evaluate.py
     ```

---

## 👥 Team

**Team name:** Timidez  
**Group:** A  
**Course:** DL + NLP (2024/2025)  
**Institution:** Universidad Pontificia Comillas (ICAI)  

---