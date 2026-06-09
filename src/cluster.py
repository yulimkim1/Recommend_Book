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
df = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))


#get book info from google books api with title 
def get_book_info(title):
    """given title, collect book's info from google books api"""
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": f"intitle:{title}",
            "key": API_KEY, 
            "maxResults": 1,
            "printType": "books",
            "langRestrict": "en"
        }
    
    response = requests.get(url, params = params)
    data = response.json()
    items = data.get("items", [])

    if not items:
        return None

    info = items[0].get("volumeInfo", {})
    book_data ={
        "title": info.get("title", "Unknown"),
        "authors": ", ".join(info.get("authors", ["Unknown"])),
        "subject": ", ".join(info.get("categories", ["Unknown"])),
        "categories": ", ".join(info.get("categories", ["Unknown"])),
        "page_count": info.get("pageCount", 0),
        "published_date": info.get("publishedDate", "Unknown"),
        "description": info.get("description", "")
    }
    return book_data

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
    subject_col = f"subject_{book_data['subject']}".lower().replace(" ", "_")
    if subject_col in vector.columns:
        vector[subject_col] = 1

    #category 
    for cat in book_data["categories"].split(","):
        cat_col = f"category_{cat.strip().lower().replace(" ", "_")}"
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

    return vector 

def get_recommendations(title, n=5):
    """takes in book title, returns n recommended books based on clustering similarity"""
    #get book info from API 
    book_info = get_book_info(title)

    #no match 
    if not book_info:
        return f"No book found with title '{title}'"
    
    #build feature vector 
    book_vector = process_book_features(book_info)
    #get cluster prediction with new features 
    cluster_id = int(kmeans.predict(book_vector)[0])

    #get books from matching predicted cluster
    #make sure it is not the same as given title 
    cluster_books = df[(df["cluster"] == cluster_id) & (df["title"].str.lower() != book_info["title"].lower())].copy()

    #return n recommendations 
    #random sample of the cluster
    # book_recommendations = cluster_books.sample(min(n, len(cluster_books)))

    #get feature matrix rows for cluster books 
    cluster_indices = cluster_books.index
    cluster_features = features.loc[cluster_indices]

    #calculate cosine similarity 
    similarities = cosine_similarity(book_vector, cluster_features)[0]

    cluster_books["sim_score"] = similarities

    #sample using cosine similarity 
    book_recommendations = cluster_books.sort_values("sim_score", ascending = False).head(n)


    return book_info, book_recommendations[["title", "authors", "subject", "categories"]]

