import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATASET_FILE = 'UpdatedResumeDataSet.csv' # Ensure this file is in the same folder
NUM_CLUSTERS = 15 # Adjust based on number of job categories

# --- STEP 1: SETUP NLTK ---
# Download necessary NLTK data for text processing
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# --- STEP 2: PREPROCESSING FUNCTION ---
def clean_resume(text):
    """
    Cleans resume text by removing URLs, special characters, and stopwords.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http/https/www) -- Added 'r' before the string
    text = re.sub(r'http\S+\s*', ' ', text)
    
    # Remove RT and cc
    text = re.sub(r'RT|cc', ' ', text)
    
    # Remove hashtags and mentions -- Added 'r'
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    
    # Remove punctuation and special characters -- Added 'r'
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    
    # Remove non-ASCII characters -- Added 'r'
    text = re.sub(r'[^\x00-\x7f]', r' ', text) 
    
    # Remove extra whitespace -- Added 'r'
    text = re.sub(r'\s+', ' ', text)
    
    # Remove Stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    
    return " ".join(filtered_text)

# --- STEP 3: MAIN EXECUTION ---
def run_resume_parser():
    print("\n--- 1. Loading Dataset ---")
    try:
        df = pd.read_csv(DATASET_FILE)
        print(f"Dataset loaded successfully: {len(df)} resumes found.")
    except FileNotFoundError:
        print(f"ERROR: '{DATASET_FILE}' not found. Please download it from Kaggle and place it in this directory.")
        return

    # Apply Cleaning
    print("Cleaning resume text...")
    df['cleaned_resume'] = df['Resume'].apply(lambda x: clean_resume(x))

    # --- STEP 4: CLUSTERING (Unsupervised Learning) ---
    print("\n--- 2. Clustering Skills ---")
    # Vectorization (TF-IDF)
    tfidf = TfidfVectorizer(max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_resume'])
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    kmeans.fit(tfidf_matrix)
    df['cluster_label'] = kmeans.labels_
    print("Clustering complete. Resumes grouped into skill categories.")

    # --- STEP 5: JOB MATCHING & SCORING ---
    print("\n--- 3. Candidate Scoring ---")
    
    # Define a Mock Job Description (You can change this to test different roles)
    job_description = """
    We are looking for a Data Scientist with strong experience in Python, Machine Learning, 
    and Deep Learning. Proficiency in libraries like Scikit-Learn, Pandas, and NumPy is essential.
    Experience with NLP and Data Visualization (Matplotlib, Tableau) is a plus.
    """
    print(f"Target Job: Data Scientist (Python/ML)")
    
    # Process Job Description
    cleaned_job_desc = clean_resume(job_description)
    job_vector = tfidf.transform([cleaned_job_desc])

    # Calculate Cosine Similarity
    similarity_scores = cosine_similarity(job_vector, tfidf_matrix)
    
    # Create 0-10 Score
    df['similarity_score'] = similarity_scores.flatten()
    df['rating_out_of_10'] = df['similarity_score'] * 10
    
    # Sort Candidates
    top_candidates = df.sort_values(by='rating_out_of_10', ascending=False).head(5)

    # --- STEP 6: DISPLAY RESULTS ---
    print("\n" + "="*50)
    print("TOP 5 CANDIDATES MATCHING THE JOB")
    print("="*50)
    
    for index, row in top_candidates.iterrows():
        print(f"Candidate ID   : {index}")
        print(f"Category       : {row['Category']}")
        print(f"Match Score    : {row['rating_out_of_10']:.2f} / 10")
        print(f"Cluster Group  : {row['cluster_label']}")
        print("-" * 30)

if __name__ == "__main__":
    run_resume_parser()
