# call google API to get book data 
# use to build books data set to train cluster model 
import requests
import os
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GOOGLE_BOOKS_API_KEY')
print("API KEY loaded:", API_KEY is not None)

def get_books_by_genre(genre, max_results = 200):
    """get books from google books api by genre"""
    """paginate since API can only pull 40 books at a time"""
    url = "https://www.googleapis.com/books/v1/volumes"
    books = []
    start_index = 0
    page_size = 40 #max per request 

    while len(books) < max_results:
        params = {"q": f"subject:{genre}",
                "key": API_KEY, 
                "maxResults": page_size,
                "startIndex": start_index,
                "printType": "books",
                "langRestrict": "en"}

        response = requests.get(url, params = params)
        data = response.json()

        items = data.get("items", [])

        #check if we get items 
        if not items: 
            break 

        #proceed if yes items 
        for book in data.get("items", []):
            info = book.get("volumeInfo", {})
            book_data = {
                "title": info.get("title", "Unknown"),
                "authors": ", ".join(info.get("authors", ["Unknown"])),
                "publisher": info.get("publisher", "Unknown"),
                "subject": genre,
                "categories": ", ".join(info.get("categories", ["Unknown"])),
                "maturityRating": info.get("maturityRating", "Unknown"),
                "page_count": info.get("pageCount", 0),
                "average_rating": info.get("averageRating", 0),
                "ratings_count": info.get("ratingsCount", 0),
                "published_date": info.get("publishedDate", "Unknown"),
                "description": info.get("description", "")
            }
            books.append(book_data)
        start_index += page_size
        time.sleep(1) #avoid API rate limits

    return books

#for cleaning up publishing date 
def normalize_date(date_str):
    """normalize the dates """
    #empty or unknown date -- NaT 
    if pd.isna(date_str) or date_str == "Unknown":
        return pd.NaT
    #if the date is just a year, add -01-01 to make it a full date
    if len(str(date_str).strip()) == 4:
        return pd.to_datetime(str(date_str) + "-01-01", errors='coerce')
    return pd.to_datetime(date_str, errors='coerce')


def build_books_dataset():
    """build the books dataset by collecting data by genre via API """
    #List of Subjects to collect data for 
    subjects = [
        "fiction",
        "historical fiction",
        "mystery",
        "science fiction",
        "fantasy",
        "romance",
        "thriller",
        "biography",
        "self help",
        "science",
        "nonfiction"
    ]

    all_books = []
    for subject in subjects: 
        print(f"getting books for genre: {subject}")
        books = get_books_by_genre(subject)
        all_books.extend(books)
        time.sleep(1) #avoid API rate limits 

    df = pd.DataFrame(all_books)

    #drop duplicates by title 
    df = df.drop_duplicates(subset = "title")

    # #drop with no page count -- potentially inaccurate data 
    df = df[df["page_count"] > 0]

    #drop academic books 
    academic_categories = ["Literary Criticism", "Reference", "Language Arts & Disciplines"]
    df = df[~df["categories"].str.contains("|".join(academic_categories), na=False)]

    #normalize published date
    df["published_date"] = df["published_date"].apply(normalize_date)

    #remove any books with NA publish date OR publish date before 1950
    df = df.dropna(subset=["published_date"])
    cutoff_date = pd.Timestamp("1990-01-01")
    df = df[df["published_date"] >= cutoff_date]

    print(f"dataset built with {len(df)} books")

    df.to_csv("data/books.csv", index = False)
    print("dataset saved as books.csv")


#actually build the dataset 
build_books_dataset()