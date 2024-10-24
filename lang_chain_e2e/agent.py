# Copyright 2024 Google, LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import getpass
import os

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a sarcastic know it all who takes pride in mixing factual answers with subtle insults"),
        MessagesPlaceholder(variable_name="messages")
    ])

def main():
    """This is the main program, a simple flow of get your api key, get the llm, and start a Q&A session,
    NOTE: there is no session history on this example."""
    llm = OllamaLLM(model="llama3.2")
    
    print("** Type 'exit' or 'quit' to end the program")
    while True:
        print("Query:", end=" ")
        try:
            line = input()
        except EOFError:
            break
        if line == 'exit' or line == 'quit':
            break
        chain = prompt | llm
        resp = chain.invoke({"messages": [HumanMessage(content=line)]})
        print(resp)