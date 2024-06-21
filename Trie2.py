import json
from datetime import datetime, timedelta

class TrieNode:
   def __init__(self):
      self.children = {}
      self.end = True
      self.books = []

class Trie:
   def __init__(self):
      self.root = TrieNode()

   def insert(self, word, book):
      node = self.root
      for char in word.lower():
         if char not in node.children:
            node.children[char] = TrieNode()
         node = node.children[char]
      node.end = False
      node.books.append(book)

   def search(self, word):
      node = self.root
      for char in word.lower():
         if char not in node.children:
            return []
         node = node.children[char]
      return  node.books

class Book:
   def __init__(self, title, author, publication_date):
      self.title = title
      self.author = author
      self.publication_date = publication_date

class User:
   def __init__(self, username, password):
      self.username = username
      self.password = password

class Library:
   def __init__(self):
      self.books = []
      self.users = []
      self.transactions = []
      self.trie = Trie()

   def add_user(self, user):
      if any(u.username == user.username for u in self.users):
         print(f'{user.username} already exists')
      else:
         self.users.append(user)
         print(f'{user.username} has been added to the system')

   def authenticate_user(self, username, password):
      user = next((u for u in self.users if u.username == username and u.password == password), None)
      if user:
         return user
      print("Invalid username or password")
      return None

   def add_book(self, book):
      self.books.append(book)
      self.trie.insert(book.title, book)

   def search_books(self, title):
      return self.trie.search(title)

   def save_to_file(self, filename):
      with open(filename, 'w') as file:
         json.dump([book.__dict__ for book in self.books], file)
      print(f"Data Saved to {filename}")

   def load_from_file(self, filename):
      with open(filename, 'r') as file:
         books_data = json.load(file)
         self.books = [Book(**data) for data in books_data]
         for book in self.books:
            self.trie.insert(book.title, book)
      print(f"Data loaded from {filename}")

# Example usage:
library = Library()
user1 = User('username1', 'password1')
library.add_user(user1)

book1 = Book('Book Title 1', 'Author 1', datetime.now())
library.add_book(book1)

library.save_to_file('library.json')

library.load_from_file('library.json')