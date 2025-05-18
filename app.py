pip install langchain openai faiss-cpu tiktoken PyMuPDF
import fitz  # PyMuPDF
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 1. Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")
    return full_text

# 2. Load the PDF and split into chunks
pdf_text = extract_text_from_pdf("path_to_your_pdf.pdf")  # Replace with the path to your PDF file

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_text(pdf_text)

# 3. Create vector store with embeddings
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(docs, embeddings)

# 4. Set up retriever
retriever = db.as_retriever()

# 5. Create RAG chain for question answering
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    retriever=retriever,
    return_source_documents=True
)

# 6. Chatbot loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = rag_chain({"query": query})
    print("\nBot:", result['result'])
