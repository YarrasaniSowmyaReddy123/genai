from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "AIzaSyDRcPvToWmH3a5jMrova8vG8t349JVvx8c"

# 1. Load and split your documents
loader = TextLoader("docs/my_knowledge.txt")  # Load your custom knowledge
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 2. Create vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# 3. Set up retriever
retriever = db.as_retriever()

# 4. Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    retriever=retriever,
    return_source_documents=True
)

# 5. Run chatbot loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = rag_chain({"query": query})
    print("\nBot:", result['result'])
