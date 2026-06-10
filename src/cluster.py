#takes in a book title and finds recommendations 

import pandas as pd
import numpy as np
import joblib 
import os
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

API_KEY = os.getenv('GOOGLE_BOOKS_API_KEY')

#saved KMeans model, scaler, and feature columns
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

kmeans = joblib.load(os.path.join(DATA_DIR, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
tfidf = joblib.load(os.path.join(DATA_DIR, "tfidf_vectorizer.pkl"))
feature_columns = joblib.load(os.path.join(DATA_DIR, "feature_columns.pkl"))
features = joblib.load(os.path.join(DATA_DIR, "features.pkl"))
feature_weights = joblib.load(os.path.join(DATA_DIR, "feature_weights.pkl"))
df = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))

#need to infer genre for given book into ones used in search when fetching data 
#use categories AND descriptions to help build 
def infer_subject(categories_str, description=""):
    text = (categories_str + " " + description).lower()
    mapping = [
        #subject                #keyword
        ("historical fiction", ["historical fiction", "historical"]),
        ("science fiction",    ["science fiction", "sci-fi", "dystopian"]),
        ("fantasy",            ["fantasy", "magic", "dragons"]),
        ("mystery",            ["mystery", "detective"]),
        ("thriller",           ["thriller", "suspense"]),
        ("romance",            ["romance", "love story"]),
        ("biography",          ["biography", "memoir", "autobiography"]),
        ("self help",          ["self-help", "self help"]),
        ("science",            ["science"]),
        ("fiction",            ["fiction", "novel"]),
    ]
        
    for subject, keywords in mapping:
        if any(k in text for k in keywords):
            return subject
    #if no other matches, non fiction -- largest range of categories 
    return "nonfiction"

#function to apply feature weights 
def apply_weights(vector, weights):
    for col in vector.columns:
        if col == "page_count_scaled":
            vector[col] *= weights["page"]
        elif col == "publish_year_scaled":
            vector[col] *= weights["publish"]
        elif col.startswith("subject_"):
            vector[col] *= weights["subject"]
        elif col.startswith("category_"):
            vector[col] *= weights["category"]
        elif col.startswith("tfidf_"):
            vector[col] *= weights["tfidf"]
    return vector

#get book info from google books api with title 
def get_book_info(title):
    """given title, collect book's info from google books api"""
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": f"intitle:{title}",
            "key": API_KEY, 
            "maxResults": 10,
            "printType": "books",
            "langRestrict": "en",
            "orderBy": 'relevance'
        }
    
    response = requests.get(url, params = params)
    data = response.json()
    items = data.get("items", [])

    if not items:
        return None

    #look for the best book with categories 
    # prefer an edition that actually has categories
    def score(info):
        cats = " ".join(info.get("categories", [])).lower()
        #ideally any edition that has more than "fiction"
        specific = any(k in cats for k in
            ["historical", "science fiction", "fantasy", "mystery", "thriller",
             "romance", "biography", "memoir", "detective", "suspense"])
        return (specific, bool(info.get("description")), bool(info.get("categories")))

    #find best edition 
    best = max((it.get("volumeInfo", {}) for it in items), key=score)

    categories = ", ".join(best.get("categories", ["Unknown"]))

    return {
        "title": best.get("title", "Unknown"),
        "authors": ", ".join(best.get("authors", ["Unknown"])),
        "subject": infer_subject(categories, best.get("description", "")),
        "categories": categories,
        "page_count": best.get("pageCount", 0),
        "published_date": best.get("publishedDate", "Unknown"),
        "description": best.get("description", ""),
    }


#process book data to match features in cluster model 
def process_book_features(book_data):
    """process book data to match features in cluster model"""
    #empty feature vector 
    vector = pd.DataFrame(0, index=[0], columns=feature_columns)

    #get publish year otherwise use median 
    try: 
        year = int(str(book_data["published_date"])[:4]) #get year 
    except:
        year = int(df["publish_year"].median()) #if not use median 

    #get page count otherwise use median 
    page_count = book_data["page_count"] if book_data["page_count"] > 0 else df["page_count"].median()

    #scale page count and publish year together 
    # scaled_values = scaler.transform([[page_count, year]])[0]
    scaled_values = scaler.transform(pd.DataFrame([[page_count, year]], columns=["page_count", "publish_year"]))[0]
    vector["page_count_scaled"] = scaled_values[0]
    vector["publish_year_scaled"] = scaled_values[1]

    #subject 
    subject_col = f"subject_{book_data['subject']}".lower().replace(' ', '_')
    if subject_col in vector.columns:
        vector[subject_col] = 1

    #category 
    for cat in book_data["categories"].split(","):
        cat_col = f"category_{cat.strip().lower().replace(' ', '_')}"
        if cat_col in vector.columns:
            vector[cat_col] = 1

    #tf-idf
    description = book_data.get("description", "")
    if description:
        tfidf_vector = tfidf.transform([description]).toarray()[0]
        tfidf_feature_names = [f"tfidf_{term}" for term in tfidf.get_feature_names_out()]
        for f_name, f_val in zip(tfidf_feature_names, tfidf_vector):
            if f_name in vector.columns:
                vector[f_name] = f_val

    #apply weights to vector 
    vector = apply_weights(vector, feature_weights)

    return vector 

def get_recommendations(title, n=5):
    #get book info from API 
    book_info = get_book_info(title)
    #no match 
    if not book_info:
        return f"No book found with title '{title}'"
    #build feature vector 
    book_vector = process_book_features(book_info)

    #use cosine similarity 
    similarities = cosine_similarity(book_vector, features)[0]
    #add in sim scores 
    df_scored = df.copy()
    df_scored["sim_score"] = similarities

    # exclude the searched book 
    df_scored = df_scored[df_scored["title"].str.lower() != book_info["title"].lower()]

    # top n by similarity
    book_recommendations = df_scored.sort_values("sim_score", ascending=False).head(n)

    return book_info, book_recommendations[["title", "authors", "subject", "categories"]]
