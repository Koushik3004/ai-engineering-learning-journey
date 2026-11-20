from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

docs=["Delhi is the capital of India","Kolkata is the capital of West Bengal","Patna is the capital of Bihar"]

result = embedding.embed_documents(docs)

print(str(result))