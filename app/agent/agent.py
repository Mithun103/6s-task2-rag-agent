"""
Provides a ReAct-style Retrieval-Augmented Generation (RAG) agent for 
document-based Q&A. Combines a language model (LLM) with vector-based 
retrieval via KnowledgeSearchTool to deliver accurate and efficient responses.

Key Features:
- Multi-step reasoning with early stopping
- Document retrieval using embeddings
- Transparent intermediate reasoning

Functions:
- create_rag_agent(llm, embedder, vector_store): Returns a configured AgentExecutor 
  for knowledge-based question answering.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from app.models.llm import get_llm
from app.config import settings
from app.agent.tools import KnowledgeSearchTool, WipeVectorStoreTool
from app.core.vector_store import VectorStore
from app.models.embedder import Embedder


def create_rag_agent(llm, embedder: Embedder, vector_store: VectorStore) -> AgentExecutor:
    """Initializes and returns a ReAct agent for document interaction.

    Creates an efficient agent that aims to provide answers in a single iteration
    when possible, using a custom prompt that encourages direct responses.

    Args:
        llm: An initialized LangChain compatible language model instance.
        embedder: An instance of the Embedder class for vectorizing queries.
        vector_store: An instance of the VectorStore class for document retrieval.

    Returns:
        An AgentExecutor instance optimized for direct responses.

    Raises:
        ValueError: If the prompt template is missing required variables.
    """
    # Initialize tools with clear, focused descriptions
    tools = [
        KnowledgeSearchTool(
            llm=llm,
            embedder=embedder,
            vector_store=vector_store
        ),
        WipeVectorStoreTool(
            vector_store=vector_store
        )
    ]

    # Custom prompt that balances efficiency with thoroughness
    prompt = PromptTemplate.from_template("""You are an intelligent assistant focused on providing complete and accurate answers efficiently.

CRITICAL FORMATTING RULES:
1. After EACH tool use, you MUST provide a Thought and Final Answer
2. ALWAYS format your response EXACTLY as shown below
3. DO NOT add any extra formatting or sections

EXACT Response Format Required:

Thought: [One sentence about what you found or did]

Final Answer: [MUST be EXACTLY one of these three formats:]

1. If information found:
Based on the documents: [Your answer]
Sources:
- Document [ID], Page [number]
- Document [ID], Page [number]

2. If no information found:
No relevant information found in the documents.

3. If database cleared:
Vector store wiped successfully.

Tools available: {tool_names}

Tool Descriptions:
{tools}

Response Format:
Question: {input}
Thought: [Brief explanation of search strategy]
Action: [Tool name]
Action Input: [Clear search parameters or None]
Observation: [Tool result]

At ANY point, if you need to give a final answer (including during iteration cutoff):
1. If you found relevant information:
   Final Answer: Based on the documents: [Concise answer including ONLY the most relevant information]
   Sources: [List ONLY the most relevant source files]

2. If you found NO relevant information:
   Final Answer: No relevant information found in the documents.

3. If you cleared the database:
   Final Answer: Vector store wiped successfully.

Note: Always prioritize giving a clear, formatted answer over continuing iterations!

Start now:

Question: {input}
{agent_scratchpad}""")

    # Create the agent with React structure and smart iteration
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Keep it at 3 iterations maximum
        early_stopping_method="force",  # Force stop after max_iterations
        return_intermediate_steps=True,  # Show thinking process
        handle_tool_error=True,  # Better error handling
        timeout=10,  # Add a timeout of 10 seconds to prevent hanging
    )

    return agent_executor
