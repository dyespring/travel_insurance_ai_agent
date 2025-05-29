from config_langchain import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config_langchain import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

#1. Define the prompt template with a variable placeholder
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate.from_template(template)

#2. Initialize the language model with your OpenAI API key
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

#3. Create an LLM chain using the prompt and the model
chain = LLMChain(llm=llm, prompt=prompt)

#4. Run the chain with input for the 'product' field
response = chain.run(product="insurance AI agents")

#5. Print the generated response
print(response)

