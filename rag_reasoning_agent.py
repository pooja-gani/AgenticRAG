import streamlit as st
from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG with Reasoning",
    page_icon=None,
    layout="wide"
)

# Main title and description
st.title("Agentic RAG with Reasoning")
st.markdown(
    """
    This app demonstrates an AI agent that:
    1. Retrieves relevant information from knowledge sources
    2. Reasons through the information step-by-step
    3. Answers your questions with citations

    Enter your API keys below to get started.
    """
)

# API keys input
st.subheader("API Keys")
col1, col2 = st.columns(2)
with col1:
    gemini_api_key = st.text_input(
        "Google (Gemini) API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Get your key from https://aistudio.google.com/apikey"
    )
with col2:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get your key from https://platform.openai.com/"
    )

# Proceed only when both keys are provided
if gemini_api_key and openai_api_key:

    # Session state: list of knowledge URLs and a set to track loaded ones
    if 'kb_urls' not in st.session_state:
        st.session_state.kb_urls = [
            "https://www.theunwindai.com/p/mcp-vs-a2a-complementing-or-supplementing"
        ]  # default source
    if 'loaded_urls' not in st.session_state:
        st.session_state.loaded_urls = set()

    # Initialize knowledge base (cached)
    @st.cache_resource(show_spinner="Loading knowledge base...")
    def init_knowledge() -> Knowledge:
        """Create and return the Knowledge object backed by LanceDB."""
        kb = Knowledge(
            vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="agno_docs",
                search_type=SearchType.vector,
                embedder=OpenAIEmbedder(api_key=openai_api_key),
            )
        )
        return kb

    # Initialize agent (cached)
    @st.cache_resource(show_spinner="Loading agent...")
    def init_agent(kb: Knowledge) -> Agent:
        """Create an Agent configured with reasoning tools."""
        return Agent(
            model=Gemini(id="gemini-2.5-flash", api_key=gemini_api_key),
            knowledge=kb,
            search_knowledge=True,
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Include sources in your response.",
                "Always search your knowledge before answering the question.",
            ],
            markdown=True,
        )

    # Load knowledge base and agent
    kb = init_knowledge()

    # Load default or new URLs once
    for url in st.session_state.kb_urls:
        if url not in st.session_state.loaded_urls:
            kb.add_content(url=url)
            st.session_state.loaded_urls.add(url)

    rag_agent = init_agent(kb)

    # Sidebar for managing knowledge sources
    with st.sidebar:
        st.header("Knowledge Sources")
        st.markdown("Add URLs to expand the knowledge base:")

        st.write("Current sources:")
        for i, url in enumerate(st.session_state.kb_urls):
            st.text(f"{i+1}. {url}")

        st.divider()
        new_url = st.text_input(
            "Add new URL",
            placeholder="https://example.com/article",
            help="Enter a URL to add to the knowledge base"
        )

        if st.button("Add URL", type="primary"):
            if new_url:
                if new_url not in st.session_state.kb_urls:
                    st.session_state.kb_urls.append(new_url)
                with st.spinner("Loading new documents..."):
                    if new_url not in st.session_state.loaded_urls:
                        kb.add_content(url=new_url)
                        st.session_state.loaded_urls.add(new_url)
                st.success(f"Added: {new_url}")
                st.rerun()
            else:
                st.error("Please enter a URL")

    # Main query section
    st.divider()
    st.subheader("Ask a Question")

    # Suggested prompts
    st.markdown("Try these prompts:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What is MCP?", use_container_width=True):
            st.session_state.query = "What is MCP (Model Context Protocol) and how does it work?"
    with col2:
        if st.button("MCP vs A2A", use_container_width=True):
            st.session_state.query = "How do MCP and A2A protocols differ, and are they complementary or competing?"
    with col3:
        if st.button("Agent Communication", use_container_width=True):
            st.session_state.query = "How do MCP and A2A work together in AI agent systems for communication and tool access?"

    # Query input box
    query = st.text_area(
        "Your question:",
        value=st.session_state.get("query", "What is the difference between MCP and A2A protocols?"),
        height=100,
        help="Ask anything about the loaded knowledge sources"
    )

    # Execute query
    if st.button("Get Answer with Reasoning", type="primary"):
        if query:
            # Create two columns for streaming: thoughts and answer
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("### Reasoning Process")
                thought_container = st.container()
                thought_placeholder = thought_container.empty()

            with col_right:
                st.markdown("### Answer")
                result_container = st.container()
                result_placeholder = result_container.empty()

            # Accumulators
            sources = []
            final_answer = ""
            thought_text = ""

            # Stream the agent's response
            with st.spinner("Searching and reasoning..."):
                for event in rag_agent.run(
                    query,
                    stream=True,
                    stream_events=True,
                ):
                    # Update reasoning (if present)
                    if hasattr(event, 'reasoning_content') and event.reasoning_content:
                        thought_text = event.reasoning_content
                        thought_placeholder.markdown(thought_text, unsafe_allow_html=True)

                    # Update answer text
                    if hasattr(event, 'content') and event.content and isinstance(event.content, str):
                        final_answer += event.content
                        result_placeholder.markdown(final_answer, unsafe_allow_html=True)

                    # Collect source URLs if present
                    if hasattr(event, 'citations') and event.citations:
                        if hasattr(event.citations, 'urls') and event.citations.urls:
                            sources = event.citations.urls

            # Display sources if any
            if sources:
                st.divider()
                st.subheader("Sources")
                for src in sources:
                    title = src.title or src.url
                    st.markdown(f"- [{title}]({src.url})")
        else:
            st.error("Please enter a question")

else:
    # Instructions shown when API keys are not provided
    st.info(
        """
        Welcome! To use this app, you need:

        1. Google API Key (for Gemini model)
           - Get it at https://aistudio.google.com/apikey

        2. OpenAI API Key (for embeddings)
           - Get it at https://platform.openai.com/

        Enter both keys above to start.
        """
    )

# Footer: explanation of components
st.divider()
with st.expander("How This Works"):
    st.markdown(
        """
        This app uses Agno to build a Q&A system:

        1. Knowledge Loading: URLs are fetched and stored in a LanceDB vector store.
        2. Vector Search: OpenAI embeddings are used for semantic search.
        3. Reasoning Tools: The agent can run step-by-step analysis.
        4. Gemini Model: Generates final answers with citations.

        Key components:
        - `Knowledge`: loads documents from URLs
        - `LanceDb`: vector database for similarity search
        - `OpenAIEmbedder`: creates embeddings
        - `ReasoningTools`: enables stepwise reasoning
        - `Agent`: coordinates retrieval and generation
        """
    )
