# call google API to get book data 
# use to build books data set to train cluster model 


import requests
import os
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GOOGLE_BOOKS_API_KEY')

def get_books_by_genre(genre, max_results = 50):
    """get books from google books api by genre"""
    url = "https://www.googleapis.com/books/v1/volumes"
    books = []

    params = {"q": f"subject:{genre}",
             "key": API_KEY, 
             "maxResults": max_results,
             "printType": "books",
             "langRestrict": "en"}

    response = requests.get(url, params = params)
    data = response.json()

    for book in data.get("items", []):
        info = book.get("volumeInfo", {})
        book = {
            "title": info.get("title", "Unknown"),
            "authors": ", ".join(info.get("authors", ["Unknown"])),
            "subject": genre,
            "categories": ", ".join(info.get("categories", ["Unknown"])),
            "page_count": info.get("pageCount", 0),
            "average_rating": info.get("averageRating", 0),
            "ratings_count": info.get("ratingsCount", 0),
            "published_date": info.get("publishedDate", "Unknown"),
            "description": info.get("description", "")
        }
        books.append(book)

    return books
        