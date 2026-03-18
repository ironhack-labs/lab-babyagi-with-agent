# FAISS-backend memory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class VectorMemory:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.texts = []
        self.store = None

    def add(self, task, result):
        text = f"Task: {task}\nResult: {result}"
        self.texts.append(text)

        self.store = FAISS.from_texts(self.texts, self.embeddings)

    def query(self, query):
        if not self.store:
            return ""

        docs = self.store.similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in docs])