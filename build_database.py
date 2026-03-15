from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

print("1. Loading the PDF...")
# This reads your PDF file
loader = PyPDFLoader("Adarsh_Kumar_Resume.pdf")
pages = loader.load()

print("2. Chopping the text into chunks...")
# This is our "Chef's knife". It chops the text into 500-character chunks.
# The "overlap" means it keeps a little bit of the previous sentence so we don't lose context mid-word!
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks!")

print("3. Turning chunks into coordinates (Embeddings) and saving to ChromaDB...")
# We use a free, fast embedding model from HuggingFace
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# This creates a local folder called "my_vector_db" and saves all the data inside it!
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model, 
    persist_directory="./my_vector_db"
)

print("✅ Success! Your Vector Database is built and ready.")