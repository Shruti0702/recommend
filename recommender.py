import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
with open('similarity_scores.pkl', 'rb') as f:
    similarity_scores = pickle.load(f)
books.drop_duplicates('Book-Title', inplace=True)
print(books.head())
def recommend(book_name):
    try:
        print("Available book titles:", books['Book-Title'].unique())
        
        if book_name not in books['Book-Title'].values:
            print(f"Book '{book_name}' not found in the dataset.")
            return []
        
        index = np.where(books['Book-Title'] == book_name)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == books['Book-Title'].iloc[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            data.append(item)
        
        print("Recommended books:", data)
        
        return data
    except IndexError:
        print("IndexError: The book might not exist in the dataset.")
        return []
