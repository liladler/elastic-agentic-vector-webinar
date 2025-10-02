import streamlit as st
import os
import asyncio
from elasticsearch import Elasticsearch
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import json
from PIL import Image
import torch
import numpy as np
from colpali_engine.models import ColPali, ColPaliProcessor
from pathlib import Path
from dotenv import load_dotenv
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Agentic Search Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Wider sidebar */
[data-testid="stSidebar"] {
    min-width: 400px;
    max-width: 400px;
}
[data-testid="stSidebar"] > div:first-child {
    width: 400px;
}

.big-font {
    font-size: 24px !important;
    font-weight: bold;
}
.tool-call {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    border-left: 3px solid #4CAF50;
    margin: 10px 0;
}
.search-result {
    background-color: #e8f4f8;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.agent-thinking {
    background-color: #fff3cd;
    padding: 10px;
    border-radius: 5px;
    border-left: 3px solid #ffc107;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'es_client' not in st.session_state:
    st.session_state.es_client = None
if 'colpali_model' not in st.session_state:
    st.session_state.colpali_model = None
if 'colpali_processor' not in st.session_state:
    st.session_state.colpali_processor = None
if 'search_agent' not in st.session_state:
    st.session_state.search_agent = None
if 'agent_history' not in st.session_state:
    st.session_state.agent_history = []
if 'workflow_container' not in st.session_state:
    st.session_state.workflow_container = None
if 'step_counter' not in st.session_state:
    st.session_state.step_counter = 0

# Configuration
INDEX_NAME = "technical_docs_webinar"
ELASTIC_URL = os.getenv("ELASTIC_URL", "")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1")

# Example questions
EXAMPLE_QUESTIONS = [
    "What is the architecture diagram of the knowledge base?",
    "Explain the relevance pyramid and its different stages",
    "How does ColPali work and why use it for image search?",
    "What is hybrid search and how does it combine different search techniques?",
    "Using my Agentic app, how could I know 'what is the temperature of widget “XYZ” after two minutes'?"
]

@st.cache_resource
def initialize_elasticsearch():
    """Initialize Elasticsearch client"""
    try:
        es_client = Elasticsearch(
            ELASTIC_URL,
            api_key=ELASTIC_API_KEY,
            verify_certs=True
        )
        if es_client.ping():
            return es_client, None
        return None, "Failed to connect to Elasticsearch"
    except Exception as e:
        return None, f"Error: {str(e)}"

@st.cache_resource
def initialize_colpali():
    """Initialize ColPali model"""
    try:
        model_name = "vidore/colpali-v1.2"
        colpali_model = ColPali.from_pretrained(model_name)
        colpali_processor = ColPaliProcessor.from_pretrained(model_name)
        return colpali_model, colpali_processor, None
    except Exception as e:
        return None, None, f"Error loading ColPali: {str(e)}"

def process_image_with_colpali(image, model, processor):
    """Process image with ColPali"""
    try:
        if isinstance(image, str):
            image = Image.open(image)
        
        batch_images = processor.process_images([image]).to(model.device)
        
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            multi_vectors = image_embeddings[0].cpu().numpy()
            avg_vector = multi_vectors.mean(axis=0)
            avg_vector_norm = np.linalg.norm(avg_vector)
            if avg_vector_norm > 0:
                avg_vector = avg_vector / avg_vector_norm
        
        return {
            "multi_vectors": multi_vectors.tolist(),
            "avg_vector": avg_vector.tolist(),
            "success": True
        }
    except Exception as e:
        return {"multi_vectors": None, "avg_vector": None, "success": False, "error": str(e)}

def create_col_pali_query_vectors(query, model, processor):
    """Generate ColPali query vectors (both multi-vectors and avg_vector)"""
    try:
        batch_queries = processor.process_queries([query]).to(model.device)
        
        with torch.no_grad():
            query_embeddings = model(**batch_queries)
            multi_vectors = query_embeddings[0].cpu().numpy()
            
            # Calculate normalized average vector for kNN retrieval
            avg_vector = multi_vectors.mean(axis=0)
            avg_vector_norm = np.linalg.norm(avg_vector)
            if avg_vector_norm > 0:
                avg_vector = avg_vector / avg_vector_norm
        
        return {
            "multi_vectors": multi_vectors.tolist(),
            "avg_vector": avg_vector.tolist()
        }
    except Exception as e:
        st.error(f"Error generating query vectors: {e}")
        return None

async def call_mcp_search(query: str, size: int = 5) -> dict:
    """Call the MCP server to perform Elasticsearch search"""
    try:
        server_params = StdioServerParameters(
            command="python3.11",
            args=["mcp_elastic_server.py"],
            cwd="./"
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Call elasticsearch_search tool
                result = await session.call_tool(
                    "elasticsearch_search",
                    arguments={"query": query, "size": size}
                )
                
                # Parse the JSON response
                response_text = result.content[0].text
                return json.loads(response_text)
                
    except Exception as e:
        return {"error": str(e)}

class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query")

class SearchTool(BaseTool):
    name: str = "search_tool"
    description: str = "Searches the Elasticsearch knowledge base using RRF hybrid search + ColPali late interaction via MCP"
    args_schema: Type[BaseModel] = SearchToolInput
    
    def _run(self, query: str) -> str:
        # Step 1: Show MCP connection
        st.session_state.step_counter += 1
        mcp_step_num = st.session_state.step_counter
        
        if st.session_state.workflow_container:
            with st.session_state.workflow_container:
                st.info(f"**Step {mcp_step_num}:** 🔌 Connecting to MCP Server\n\nInitiating Model Context Protocol connection...")
        
        # Step 2: Show search execution
        st.session_state.step_counter += 1
        search_step_num = st.session_state.step_counter
        
        if st.session_state.workflow_container:
            with st.session_state.workflow_container:
                st.info(f"**Step {search_step_num}:** 🔎 Calling Elasticsearch Search via MCP\n\nSearching for: *\"{query}\"*")
        
        st.session_state.agent_history.append({
            "type": "tool_call",
            "tool": "search_tool",
            "query": query,
            "step": search_step_num,
            "mcp_step": mcp_step_num
        })
        
        try:
            # Call MCP server synchronously (Streamlit doesn't support async directly)
            search_data = asyncio.run(call_mcp_search(query))
            
            if "error" in search_data:
                return f"Error performing search via MCP: {search_data['error']}"
            
            # Process MCP results
            results = search_data.get('results', [])
            
            # Convert MCP format to expected format
            formatted_results = []
            for res in results:
                formatted_results.append({
                    "title": res.get('title', ''),
                    "content": res.get('slide_text', ''),
                    "image_path": res.get('image_path', ''),
                    "page_number": res.get('slide_number', ''),
                    "score": res.get('score', 0.0)
                })
            
            st.session_state.agent_history.append({
                "type": "tool_result",
                "tool": "search_tool",
                "results": formatted_results
            })
            
            if not formatted_results:
                return "No results found for the given query."
            
            output = f"Found {len(formatted_results)} relevant slides:\n\n"
            for i, result in enumerate(formatted_results, 1):
                output += f"**Result {i} (Slide {result['page_number']}, Score: {result['score']:.2f})**\n"
                output += f"Title: {result['title']}\n"
                output += f"Content Preview: {result['content'][:300]}...\n"
                output += f"Image: {result['image_path']}\n"
                output += "-" * 80 + "\n"
            
            return output
            
        except Exception as e:
            return f"Error performing search: {str(e)}"

class ImageAnalysisToolInput(BaseModel):
    """Input schema for ImageAnalysisTool"""
    image_url: str = Field(..., description="The path or URL of the image to analyze")
    question: str = Field(..., description="The specific question about the image")

class ImageAnalysisTool(BaseTool):
    name: str = "image_analysis_tool"
    description: str = (
        "Expert in visual content analysis, object detection, and deep image understanding. "
        "Analyzes slide images to extract information from diagrams, charts, and visual elements."
    )
    args_schema: Type[BaseModel] = ImageAnalysisToolInput
    
    def _run(self, image_url: str, question: str) -> str:
        # Increment step counter
        st.session_state.step_counter += 1
        step_num = st.session_state.step_counter
        
        # Show real-time update
        if st.session_state.workflow_container:
            with st.session_state.workflow_container:
                st.info(f"**Step {step_num}:** 🖼️ Calling Image Analysis Tool\n\nAnalyzing: *{Path(image_url).name}*")
        
        st.session_state.agent_history.append({
            "type": "tool_call",
            "tool": "image_analysis_tool",
            "image_path": image_url,
            "question": question,
            "step": step_num
        })
        
        try:
            image = Image.open(image_url)
            
                        
            st.session_state.agent_history.append({
                "type": "tool_result",
                "tool": "image_analysis_tool",
                "image_path": image_url
            })
            
            return image_url
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

def initialize_agent():
    """Initialize the search agent"""
    if st.session_state.search_agent is not None:
        return st.session_state.search_agent
    
    search_tool = SearchTool()
    image_analysis_tool = ImageAnalysisTool()
    
    agent_llm = LLM(
        model=OPENAI_MODEL_NAME,
        base_url=OPENAI_BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    search_agent = Agent(
        role="Intelligent Search Agent",
        goal="Process user questions by searching knowledge bases and analyzing slide images for visual insights",
        backstory="""
        You are an intelligent search agent that combines textual search with visual analysis. 
        You understand that slides often contain important diagrams, charts, and visual layouts 
        that provide crucial context beyond just the text.
        
        Your workflow:
        1. Use search_tool to find relevant slides
        2. Use image_analysis_tool on the top results to extract visual insights
        3. Combine both text and visual information in your answer
        
        Visual analysis helps you understand diagrams, architecture, workflows, and data 
        visualizations that are critical to answering questions accurately.
        """,
        tools=[search_tool, image_analysis_tool],
        llm=agent_llm,
        verbose=True,
        allow_delegation=False
    )
    
    st.session_state.search_agent = search_agent
    return search_agent

def ask_agent(question: str):
    """Ask a question to the agent - simplified version"""
    agent = initialize_agent()
    
    search_task = Task(
        description=f"""
        Answer the following user question by searching the knowledge base:
        
        **Question:** {question}
        
        **Instructions:**
        1. Use search_tool to find relevant slides
        2. For the top 2-3 most relevant slides, use image_analysis_tool to load their images 
        3. Synthesize all information (text + visual analysis) into a clear, comprehensive answer
        4. Start with your answer immediately - DO NOT dump raw slide content
        5. Reference slide numbers inline (e.g., "According to Slide 14...")
        6. Keep your answer focused and well-organized with bullet points or paragraphs
        
        **Special Instructions for Architecture/Diagram Questions:**
        - ONLY create a Mermaid diagram if the user's question EXPLICITLY asks for:
          * "architecture" or "system architecture"
          * "workflow" or "process flow"
          * "system design" or "how the system works"
          * A "diagram" or "visualization" of a system/process
        - If creating a diagram:
          * Place the diagram FIRST, before your explanation
          * Use this exact format:
            ```mermaid
            graph TD
                A[Component] --> B[Another Component]
            ```
          * Then provide your textual explanation below the diagram
        - For other questions (facts, definitions, specific values), just provide a text answer WITHOUT diagrams
        
        **Format:**
        - Start with a direct answer to the question (and Mermaid diagram if applicable)
        - Use inline citations like "Slide 14" or "As shown in Slide 7..."
        - DO NOT copy-paste large blocks of raw slide text
        - End with a brief "References:" line listing the slides used
        - DO NOT use '(' or ')' as it won't parse correctly
        """,
        expected_output="A clear, synthesized answer combining text and visual insights, with inline slide references. Include Mermaid diagrams ONLY if the user explicitly asks for architecture, workflow, or system design visualization.",
        agent=agent,
        multimodal=True
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[search_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return result

# Sidebar
with st.sidebar:
    st.title("🔍 Agentic Search Demo")
    st.markdown("---")
    
    st.subheader("📚 Example Questions")
    for i, question in enumerate(EXAMPLE_QUESTIONS, 1):
        if st.button(f"{i}. {question}", key=f"q{i}"):
            st.session_state.current_question = question
            st.rerun()
    
    st.markdown("---")
    st.subheader("⚙️ System Status")
    
    # Initialize components
    if st.session_state.es_client is None:
        with st.spinner("Connecting to Elasticsearch..."):
            es_client, error = initialize_elasticsearch()
            if es_client:
                st.session_state.es_client = es_client
                st.success("✓ Elasticsearch connected")
            else:
                st.error(f"✗ Elasticsearch: {error}")
    else:
        st.success("✓ Elasticsearch connected")
    
    if st.session_state.colpali_model is None:
        with st.spinner("Loading ColPali model..."):
            model, processor, error = initialize_colpali()
            if model:
                st.session_state.colpali_model = model
                st.session_state.colpali_processor = processor
                st.success("✓ ColPali loaded")
            else:
                st.error(f"✗ ColPali: {error}")
    else:
        st.success("✓ ColPali loaded")
    
    # Index stats
    if st.session_state.es_client:
        try:
            stats = st.session_state.es_client.count(index=INDEX_NAME)
            st.info(f"📊 Documents indexed: {stats['count']}")
        except:
            st.warning("⚠️ Could not fetch index stats")

# Main content
st.title("🤖 Agentic Search with Elastic Vector Database")
st.markdown("### Ask questions about the webinar content using intelligent agents!")

# Query input
# Initialize question in session state if not present
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

user_question = st.text_area(
    "Enter your question:", 
    value=st.session_state.current_question,
    height=100
)

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("🔍 Ask Agent", type="primary")
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.agent_history = []
        st.rerun()

# Execute search
if search_button and user_question:
    st.session_state.agent_history = []
    st.session_state.current_question = user_question
    st.session_state.step_counter = 0
    
    # Check if components are initialized
    if st.session_state.es_client is None:
        st.error("⚠️ Elasticsearch not connected. Please check your .env file.")
        st.stop()
    
    if st.session_state.colpali_model is None:
        st.error("⚠️ ColPali model not loaded. Please wait for it to load in the sidebar.")
        st.stop()
    
    # Create live workflow display section
    st.markdown("---")
    st.markdown("### 🔄 Agent Workflow")
    
    # Create a container for real-time updates
    workflow_container = st.container()
    st.session_state.workflow_container = workflow_container
    
    # Components are ready, run agent
    with st.spinner("🤖 Agent is processing..."):
        try:
            result = ask_agent(user_question)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.stop()
    
    # Show final step
    with workflow_container:
        st.success(f"**Step {st.session_state.step_counter + 1}:** 💡 Generated Final Answer")
    
    # Display result
    st.markdown("---")
    st.markdown("### 💡 Answer")
    
    # Extract clean answer text
    answer_text = str(result)
    if hasattr(result, 'raw'):
        answer_text = result.raw
    
    # Handle Mermaid diagrams by extracting and rendering separately
    import re
    mermaid_pattern = r'```mermaid\s*\n(.*?)\n\s*```'
    
    # Check if there are mermaid diagrams
    if '```mermaid' in answer_text:
        parts = re.split(mermaid_pattern, answer_text, flags=re.DOTALL)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            if i % 2 == 0:
                # Regular text (even indices)
                if part:
                    st.markdown(part)
            else:
                # Mermaid diagram (odd indices)
                st.subheader("📊 Architecture Diagram")
                
                # Use components.html for better rendering
                from streamlit.components.v1 import html
                
                mermaid_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                    <script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
                </head>
                <body>
                    <div class="mermaid">
                    {part}
                    </div>
                </body>
                </html>
                """
                html(mermaid_html, height=400, scrolling=True)
    else:
        # No mermaid diagrams, display normally
        st.markdown(answer_text)
    
    # Show individual slides with images
    if st.session_state.agent_history:
        with st.expander("📄 View Source Slides (with images)", expanded=False):
            for event in st.session_state.agent_history:
                if event["type"] == "tool_result" and event["tool"] == "search_tool" and "results" in event:
                    for i, res in enumerate(event["results"][:3], 1):
                        st.markdown(f"### Slide {res['page_number']}: {res['title']}")
                        
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            if res['image_path'] and Path(res['image_path']).exists():
                                st.image(res['image_path'], width='stretch')
                                st.caption(f"Relevance Score: {res['score']:.2f}")
                        with col2:
                            # Show a preview of the content
                            content_preview = res['content'][:400]
                            st.markdown(f"**Content:**")
                            st.text(content_preview + "..." if len(res['content']) > 400 else content_preview)
                        
                        st.markdown("---")

elif search_button:
    st.warning("⚠️ Please enter a question first!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<small>
Built with Elasticsearch, ColPali, and CrewAI<br>
Following the architecture from: <a href='https://www.elastic.co/search-labs/blog/late-interaction-model-colpali-scale'>Elastic ColPali Blog</a>
</small>
</div>
""", unsafe_allow_html=True) 