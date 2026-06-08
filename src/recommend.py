#gives recommendation via CLI 

import argparse 
from src.cluster import get_recommendations

def main():
    #title
    parser = argparse.ArgumentParser(description = "book recommender - find a similar book!")
    #get title
    parser.add_argument("title", type=str, help="title of the book you want to find something similar of")
    #get num recommendations 
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="number of recommendations to return (default = 5)"
    )

    args = parser.parse_args()

    print(f"finding books similar to '{args.title}' \n")

    #GET RECOMMENDATION 
    book_info, recommendations = get_recommendations(args.title, n=args.n)

    #no book found 
    if book_info is None:
        print(recommendations) #error message 
        return 

    #yes recommendation 
    print(f"you are looking for a book similar to {book_info['title']} by {book_info['authors']}")
    print('recommended books: \n')

    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}, {row['title']} by {row['authors']}, genre: {row['subject']}")

if __name__ == "__main__":
    main()



