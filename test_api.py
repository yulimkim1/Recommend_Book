import requests
import os 
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GOOGLE_BOOKS_API_KEY')

def search_books(title):
    """test google books API call"""
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": title, "key": API_KEY, "maxResults": 5}
    response = requests.get(url, params = params)
    data = response.json()

    for item in data.get("items", []):
        book_info = item["volumeInfo"]
        print(f"{book_info.get('title')} by {book_info.get('authors', ['Unknown'])}")

search_books("Harry Potter")
