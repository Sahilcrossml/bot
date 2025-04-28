from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyCxQOEEj1YWE7SaP1tMG3Ug_jwqYPIJpZI",
    streaming=True)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_main_directory = "chroma_store"
template = """Answer the question using only the context below.
If the context does not contain the answer, just respond with "Unfortunately, I'm not able to answer your query.". Do not explain or expand.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)
temps_directory = "temps"
os.makedirs(temps_directory, exist_ok=True)


def store_exists(store_name):
    store_path = os.path.join(persist_main_directory, store_name)
    return not os.path.exists(store_path)

def get_store_path(store_name):
    return os.path.join(persist_main_directory, store_name)

def text_file(filename, data):
    filepath = os.path.join(temps_directory, f'{filename}.txt')
    with open(filepath, 'w') as f:
        f.write(data)
    return filepath

def store_vectors(text_data, store_name):
    textfile = text_file(store_name, text_data)
    loader = TextLoader(textfile)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    persist_directory = get_store_path(store_name)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    os.remove(textfile)

def load_vectors(store_name):
    persist_directory = get_store_path(store_name)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectorstore

def bot_answer(name, question):
    vectors = load_vectors(name)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectors.as_retriever(),
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
        return_source_documents=False
    )
    for chunk in chain.stream(question):
        yield chunk


if __name__ == '__main__':
    # Create a new bot
    text_data = '''In a forgotten village tucked between two mountains, there was a single lamp that never went out. No one knew who lit it first, but it burned through storms, winters, and even wars.

One evening, a little girl named Lila, curious and bold, asked the village elder, "What happens if it goes out?"

The elder only smiled and said, "It won't... as long as someone cares."

That night, the wind howled harder than ever before. The villagers stayed inside, frightened. But Lila, clutching a small lantern of her own, ran into the storm. She found the great lamp flickering, almost dying.

With her tiny lantern, she shielded the flame, whispering, "Stay."

The storm passed. The lamp still burned. And from then on, every night, a different villager would visit it, just to sit by its side for a while — caring.

And the light never went out again.'''

    name = 'ai_bot'
    store_vectors(text_data, name)
    
    # Bot answer
    # question = 'Who was the brave little girl in the story?'
    # ans_heading = False
    # for chunk in bot_answer(name, question):
    #     if not ans_heading:
    #         ans_heading = True
    #         print('\nAnswer:')
    #     print(chunk.get("result", ""), flush=True)
