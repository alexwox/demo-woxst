from __future__ import annotations as _annotations

# Standard library imports
import asyncio
import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

# Third-party imports
from dotenv import load_dotenv
import logfire
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pydantic import BaseModel, Field
import streamlit as st
from tavily import AsyncTavilyClient

# Local/pydantic-ai imports
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import Message, UserPrompt, ModelTextResponse

# Environment setup
load_dotenv()
logfire.configure()
tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"]) 

# Data classes
@dataclass
class SearchDataclass:
    max_results: int
    todays_date: str

@dataclass
class ToolkitInfo:
    pdf_content: str
    source: str

@dataclass
class Deps:
    todays_date: str

class ResearchResult(BaseModel):
    research_title: str = Field(
        description='Top level Markdown heading for the query topic (prefix with #). Can be empty if insufficient information.'
    )
    research_main: str = Field(
        description='Main section providing answers for the query and research'
    )
    research_bullets: str = Field(
        description='Bullet points summarizing the answers. Can be empty if insufficient information.'
    )

# Agent configuration
SYSTEM_PROMPT = """You are a helpful research assistant and expert in research.

You have access to a toolkit of PDFs that you can use to answer the user's question.
Always call the tool before searching.

When given a question, if you need external sources, write strong keywords to do as many 
searches as needed (each with a query_number) and then combine the results. 
Never make up information.

Answer in markdown format when needed."""

search_agent = Agent(
    'openai:gpt-4',
    deps_type=Deps,
    result_type=ResearchResult,
    system_prompt=SYSTEM_PROMPT,
    retries=5
)

# Agent tools
@search_agent.tool
async def add_toolkit_information(ctx: RunContext[ToolkitInfo], query: str) -> dict[str, Any]:
    """Get information from toolkit PDFs based on the query.
    
    Args:
        query: The information to search for in the toolkit.
    Returns:
        Dictionary containing relevant text and source information.
    """
    documents = SimpleDirectoryReader("custom_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)
    return response

@search_agent.tool
async def get_search(ctx: RunContext[SearchDataclass], query: str, query_number: int) -> dict[str, Any]:
    """Get search results for a keyword query.
    
    Args:
        query: Keywords to search.
        query_number: Sequential number of the query.
    Returns:
        Dictionary containing search results.
    """    
    print(f"Search query {query_number}: {query}")
    return await tavily_client.get_search_context(query=query, max_results=3)

# Streamlit UI
def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_message_history():
    message_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            message_history.append(UserPrompt(content=msg["content"]))
        elif msg["role"] == "assistant":
            message_history.append(ModelTextResponse(content=msg["content"]))
    return message_history

def main():
    st.title("Research Assistant")
    initialize_chat()
    display_chat_history()

    if prompt := st.chat_input("What would you like to research?"):
        st.chat_message("user").markdown(prompt)
        
        assistant_container = st.chat_message("assistant")
        with assistant_container:
            message_placeholder = st.empty()
            message_placeholder.markdown("Starting research...")
            
            try:
                with st.spinner("Researching... 🔍"):
                    current_date = datetime.date.today()
                    deps = Deps(todays_date=current_date.strftime("%Y-%m-%d"))
                    message_history = get_message_history()

                    result = asyncio.run(
                        search_agent.run(
                            prompt, 
                            deps=deps,
                            message_history=message_history
                        )
                    )

                    response_parts = [
                        result.data.research_title,
                        "\n\n",
                        result.data.research_main,
                        "\n\n",
                        result.data.research_bullets
                    ]

                    full_response = ""
                    for part in response_parts:
                        full_response += part
                        message_placeholder.markdown(full_response)
                        time.sleep(0.05)
                    
                    st.session_state.messages.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": full_response}
                    ])
                    
            except Exception as e:
                message_placeholder.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()