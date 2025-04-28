"""
Absolutely! Let's build a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain** and **Chroma** to enhance your SQL query enhancement tool. This approach will allow you to:

1. **Retrieve** the exact SQL query from your `.sql` files based on user input.
2. **Enhance** the retrieved SQL query using an LLM (e.g., OpenAI's GPT-4) when the user requests improvements.


"""
## 🔧 Prerequisites
"""
Ensure you have the following Python packages installed:

```bash
pip install langchain openai chromadb
```

Additionally, you'll need an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

---
"""
## 🧱 Step-by-Step Implementation

### 1. **Load and Index SQL Files**

##We'll load your `.sql` files, split them into manageable chunks, and store them in a Chroma vector store for efficient retrieval.


import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load SQL files
loader = DirectoryLoader('sql_folder', glob='*.sql')
docs = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embeddings)


### 2. **Set Up the Retrieval and Generation Chain**

#Define a prompt template for the LLM and set up the retrieval-augmented generation chain.


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Define the prompt template
prompt_template = """
You are an expert SQL developer. Here is a SQL query:

{query}

Please suggest any improvements or optimizations.
"""
prompt = PromptTemplate(input_variables=["query"], template=prompt_template)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Set up the retrieval-augmented generation chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)


### 3. **Define the User Interaction Loop**

#Implement a loop to handle user queries and provide SQL enhancements upon request.


def get_sql_query(user_input):
    # Retrieve the relevant SQL query
    result = qa_chain.run(user_input)
    return result['result']

def enhance_sql_query(sql_query):
    # Enhance the SQL query using the LLM
    result = qa_chain.run(f"Enhance the following SQL query:\n{sql_query}")
    return result['result']

# Main interaction loop
if __name__ == "__main__":
    while True:
        user_input = input("Ask for a SQL query or type 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            break
        sql_query = get_sql_query(user_input)
        print(f"SQL Query: {sql_query}")
        
        enhance = input("Would you like to enhance this query? (yes/no): ").strip().lower()
        if enhance == 'yes':
            enhanced_query = enhance_sql_query(sql_query)
            print(f"Enhanced SQL Query: {enhanced_query}")


## ✅ How It Works
"""
1. **Loading and Indexing**: The SQL files are loaded and split into chunks, which are then embedded and stored in a Chroma vector store.
2. **Retrieval**: When the user asks for a SQL query, the system retrieves the most relevant chunk from the vector store.
3. **Enhancement**: If the user requests enhancements, the retrieved SQL query is passed to the LLM, which suggests improvements.

"""

## 📚 Further Reading

"""
For more detailed information on building RAG applications with LangChain, refer to the official LangChain documentation:

- [Retrieval Augmented Generation (RAG) | LangChain](https://python.langchain.com/docs/tutorials/rag/)
- [Build a Retrieval Augmented Generation (RAG) App](https://python.langchain.com/docs/tutorials/rag/)

"""

#Would you like assistance in deploying this as a web application or integrating it with a database? 