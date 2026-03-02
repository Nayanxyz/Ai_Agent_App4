from dotenv import load_dotenv                                                   #dotenv is a library to read .env folder

import os                                                                        #os library is a bridge to operating system

from langchain_core.messages import HumanMessage, AIMessage                     # Langchain are nerves connecting brain and (LLM) and parts (Todoist app)
                                                                                # messages are objects used in prompts and chat conversations.
                                                                                #HumanMessages are messages that are passed in from a human to the model.
                                                                                #AIMessage is returned from a chat model as a response to a prompt

from langchain_core.output_parsers import StrOutputParser                      #In LangChain, an Output Parser is a dedicated class designed to take the raw,
                                                                               # often unstructured text output from a Large Language Model (LLM) and
                                                                               # convert it into a structured, reliable, and usable format (such as JSON, Pydantic objects, or Python lists)
                                                                               #StrOutputParser that parses LLMResult into the top likely string.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder    #prompts is the input to the model
                                                                              #ChatPromptTemplate Use to create flexible templated prompts for chat models
                                                                              #Prompt template that assumes variable is already list of messages.
                                                                              #A placeholder which can be used to pass in a list of messages.

from langchain_google_genai import ChatGoogleGenerativeAI                     #This module integrates Google's Generative AI models, specifically the Gemini series, with the LangChain framework.
                                                                              # It provides classes for interacting with chat models and generating embeddings
                                                                              #The ``ChatGoogleGenerativeAI`` class is the primary interface for interacting with Google's Gemini chat models.
                                                                              # It allows users to send and receive messages using a specified Gemini model, suitable for various conversational AI applications.

from pydantic_core.core_schema import model_field                             #pydantic_core is the underlying high-performance Rust-based validation logic engine for the Python Pydantic library.
                                                                              #core_schema module in the Pydantic library contains the definitions used to build schemas that Pydantic's core validation and serialization logic
                                                                              #model_field It holds all the necessary information for validating and serializing that specific field,
                                                                              # including its type, constraints, default value, and any other metadata

from langchain.tools import tool                                              #tool are classes that an Agent uses to interact with the world.Each tool has a **description**.
                                                                              # Agent uses the description to choose the right tool for the job.
                                                                              #Make tools out of functions, can be used with or without arguments

from langchain.agents import create_openai_tools_agent , AgentExecutor        #Agents is a class that uses an LLM to choose a sequence of actions to take
                                                                              #In Chains, a sequence of actions is hardcoded. In Agents,a language model is used as a reasoning engine
                                                                              # to determine which actions to take and in which order
                                                                              #create_openai_tools_agent create an agent that uses OpenAI tools (gets three arguments)
                                                                              #llm: LLM to use as the agent.
                                                                              #tools: Tools this agent has access to.
                                                                              # prompt: The prompt to use. See Prompt section below for more on the expected input variables
                                                                              #AgentExecutor Agent that is using tools.

from todoist_api_python.api import TodoistAPI


load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task, desc=None):
    """Add a new task to the user's task list. Use this when the user wants to add or create a task"""
    todoist.add_task(content=task,
                     description=desc)

@tool
def show_tasks():
    """show all tasks from Todoist inbox only . Use this tool when the user wants to see their tasks. """
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks

tools = [add_task, show_tasks]

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature=0.3
)

system_prompt = """You are a helpful assistant. 
you will help the user add tasks . if the user asks to show tasks , print tasks from inbox only in a bullet list format"""

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

#chain = prompt | llm | StrOutputParser()                                                 #prompt inside llm converts String output and annalize  Chain method
agent = create_openai_tools_agent(llm, tools , prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

#response = chain.invoke({"input":user_input})

history = []
while True:
    user_input = input("You: ")
    response = agent_executor.invoke({"input": user_input, "history": history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))