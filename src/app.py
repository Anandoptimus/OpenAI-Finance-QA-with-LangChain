from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import langchain


loader = WebBaseLoader("https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html")
docs = loader.load()


llm  = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.5)

splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200)

chunk = splitter.split_documents(docs)
chunk

embeddings = OpenAIEmbeddings()

vector = FAISS.from_documents(chunk, embeddings)


chain = RetrievalQAWithSourcesChain.from_llm(
                retriever = vector.as_retriever(),
                llm = llm
)

query = "TTMT’s consolidated revenue for Q1FY24 was ?"

# answer: INR 10,22,361 mn, a growth of 42.1% YoY but a decline of 3.5% QoQ

langchain.debug = True

chain({"question": query}, return_only_outputs =True)
