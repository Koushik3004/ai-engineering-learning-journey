from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI(model="gpt-4.1-mini")

result=llm.invoke("what is capital of India")

print(result.content)