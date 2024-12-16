TASK 2:  Chat with Website Using RAG Pipeline

To implement a Retrieval-Augmented Generation (RAG) pipeline to interact with structured and unstructured data from websites, considert the below code 

1. Installation of Required Libraries:
pip install requests beautifulsoup4 sentence-transformers faiss-cpu openai

2. Data Ingestion:
Step 2.1: Crawling and Scraping Content
import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the web page.")
        return None
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()  # You can customize this to get specific elements

# Example usage for each website
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/",
    "https://und.edu/"
]

website_contents = {}
for url in urls:
    website_contents[url] = scrape_website(url)

Step 2.2: Chunking and Creating Embeddings

from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_and_embed(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = model.encode(chunks)

    return chunks, embeddings

# Dictionary to hold chunks and embeddings
embeddings_dict = {}
for url, content in website_contents.items():
    if content:  # Ensure content is not None
        chunks, embeddings = chunk_and_embed(content)
        embeddings_dict[url] = {'chunks': chunks, 'embeddings': embeddings}
      
Step 3: Store Embeddings in a Vector Database

import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Create a flat index for L2 distance
    index.add(np.array(embeddings).astype('float32'))  # Add embeddings to the index
    return index

# Create FAISS indices
faiss_indices = {}
for url, data in embeddings_dict.items():
    index = create_faiss_index(data['embeddings'])
    faiss_indices[url] = index
  
4. Query Handling
To handle user queries, we need to convert the query into embeddings, perform similarity searches, and retrieve relevant chunks.

def query_embedding(query):
    return model.encode([query])

def search_index(index, query_embedding, k=5):
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return distances, indices
Step 5: Generate Response
Using the retrieved chunks from the vector database, generate a response.

def generate_response(relevant_chunks):
    # Here you can integrate with an LLM like OpenAI's ChatGPT
    # Example of using OpenAI API (Assuming you have set your API key)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or any other model you want to use
        messages=[{"role": "user", "content": " ".join(relevant_chunks)}]
    )
    return response['choices'][0]['message']['content']

# Example of processing a query
query_text = "What are the key academic offerings at Stanford University?"
query_vec = query_embedding(query_text)

# Assuming querying Stanford's index
index = faiss_indices['https://www.stanford.edu/']
_, indices = search_index(index, query_vec)

# Retrieve relevant chunks for response generation
relevant_chunks = [embeddings_dict['https://www.stanford.edu/']['chunks'][i] for i in indices.flatten()]

final_response = generate_response(relevant_chunks)
print(final_response)
