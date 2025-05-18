from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Load RAG components: Tokenizer, Retriever, Model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Function to create FAISS index (optional, if using custom dataset)
def create_faiss_index():
    retriever.index.set_faiss_index(torch.randn(512, 768))  # Random for illustration

create_faiss_index()

# Chatbot function using RAG
def rag_chatbot(input_text: str):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # Tokenize user input
    retriever_outputs = retriever(input_ids=input_ids, return_tensors="pt")  # Retrieve relevant documents
    generated_ids = model.generate(
        input_ids=input_ids, 
        context_input_ids=retriever_outputs.context_input_ids,
        context_attention_mask=retriever_outputs.context_attention_mask,
        max_length=200
    )  # Generate response
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Running the chatbot
if __name__ == "__main__":
    print("Hello! I'm a RAG-based chatbot. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"Bot: {rag_chatbot(user_input)}")
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Load RAG components: Tokenizer, Retriever, Model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Function to create FAISS index (optional, if using custom dataset)
def create_faiss_index():
    retriever.index.set_faiss_index(torch.randn(512, 768))  # Random for illustration

create_faiss_index()

# Chatbot function using RAG
def rag_chatbot(input_text: str):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # Tokenize user input
    retriever_outputs = retriever(input_ids=input_ids, return_tensors="pt")  # Retrieve relevant documents
    generated_ids = model.generate(
        input_ids=input_ids, 
        context_input_ids=retriever_outputs.context_input_ids,
        context_attention_mask=retriever_outputs.context_attention_mask,
        max_length=200
    )  # Generate response
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Running the chatbot
if __name__ == "__main__":
    print("Hello! I'm a RAG-based chatbot. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"Bot: {rag_chatbot(user_input)}")
