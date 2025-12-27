# 📄 AI Resume Parser

> **Transforming recruitment with intelligent context-aware analysis.**

## 📌 Project Overview
The **AI Resume Parser** is a Python-based tool designed to solve the challenge of manual resume screening. Unlike traditional systems that rely on simple keyword matching (which can be easily "gamed" by buzzwords), this project uses **Natural Language Processing (NLP)** and **Machine Learning** to understand the **context** of skills and qualifications.

By implementing **Clustering Analysis** and **Cosine Similarity**, the system groups related skills and assigns a relevance score (0-10) to help recruiters identify truly qualified candidates efficiently.

## 🚀 Key Features
* **Contextual Understanding:** Goes beyond keywords to understand that terms like "managed team" and "supervised staff" are semantically related.
* **Intelligent Clustering:** Uses **K-Means Clustering** to group resumes into distinct skill categories automatically.
* **Candidate Scoring:** Ranks candidates on a scale of **0-10** based on how well they match the specific job description.
* **Efficiency:** Designed to reduce manual screening time significantly.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** NLTK (Text Preprocessing), Scikit-Learn (Clustering & Scoring), Pandas, NumPy.
* **Algorithm:** TF-IDF Vectorization, K-Means Clustering, Cosine Similarity.

## 📂 Dataset
This project uses the **Resume Dataset** from Kaggle, which contains text-based resume information and job categories.
**Note: The dataset file is not included in this repository to save space.**

## ⚙️ Installation & Setup

### 1. Clone the Repository
bash:
git clone [https://github.com/SayanModak/AI-Resume-Parser.git](https://github.com/SayanModak/AI-Resume-Parser.git)
cd AI-Resume-Parser

### 2. Install Dependencies

bash:
pip install -r requirements.txt

### 3. Download the Dataset

1. Go to [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset).
2. Download the dataset (file is named `UpdatedResumeDataSet.csv`).
3. Place the CSV file inside the project folder.

## 🏃‍♂️ Usage

Run the main script to process resumes and see the ranking results:

bash:
python main.py


**What happens next?**

1. The script loads and cleans the resumes.
2. It clusters them into skill groups.
3. It compares them against a target Job Description (e.g., "Data Scientist").
4. It prints the **Top 5 Candidates** with their match scores.

## 📊 Results

* **Context Accuracy:** Improved candidate matching over standard keyword searches.
* **Time Saved:** Reduces manual review time by automating the initial screening process.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
-------------------------------------------------------------------------------------------------------------------------------

**Submitted by Sayan Modak as part of the Internship Program at AENEXZ TECH PVT LTD.**
