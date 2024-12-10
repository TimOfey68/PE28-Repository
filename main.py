
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Настройка векторной базы
texts = [
    "Привет, друзья! Сегодня поговорим о технологиях будущего.",
    "Искусственный интеллект уже меняет нашу жизнь.",
    "Квантовые компьютеры — это революция в обработке данных."
]
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_texts(texts, embeddings)

# Настройка цепочки RetrievalQA
llm = ChatOpenAI(model="gpt-4")
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=False
)

# Выполнение запроса
query = "Как технологии изменят будущее?"
result = retrieval_chain.run(query)

# Вывод результата
print("Сгенерированный пост:")
print(result)
