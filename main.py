from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
resp = llm.invoke("Hello, world!")
print(resp)
