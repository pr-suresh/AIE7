# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph-based project that demonstrates building and serving agentic AI workflows. It contains two main graph implementations:
1. Simple agent with tool usage (`simple_agent`)
2. Agent with helpfulness evaluation loop (`agent_with_helpfulness`)

## Essential Commands

### Development
```bash
# Start local development server
uv run langgraph dev

# Test the served graph
uv run test_served_graph.py
```

### Environment Setup
1. Copy `.env.example` to `.env`
2. Add required API keys:
   - `OPENAI_API_KEY` (required)
   - `TAVILY_API_KEY` (required for search functionality)
   - `LANGSMITH_API_KEY` (optional, for tracing)

## Architecture & Key Components

### Graph Structure
The project uses LangGraph's declarative graph pattern where:
- **Nodes**: Functions that process state (e.g., `call_tool`, `call_model`)
- **Edges**: Define flow between nodes, including conditional routing
- **State**: Shared data structure (`GraphState`) containing messages and tool outputs

### Core Modules
- `app/state.py`: Defines the shared `GraphState` for all agents
- `app/tools.py`: Aggregates tools (Tavily search, Arxiv, RAG) using `ToolNode`
- `app/models.py`: Configures LLM clients with structured output support
- `app/rag.py`: Implements RAG pipeline with Qdrant vector store

### Graph Implementations
1. **Simple Agent** (`graphs/simple_agent.py`):
   - Basic tool-calling loop: agent → tools → agent
   - Uses conditional routing based on tool calls

2. **Agent with Helpfulness** (`graphs/agent_with_helpfulness.py`):
   - Adds helpfulness evaluation after responses
   - Implements retry logic if response isn't helpful
   - Uses structured output (`HelpfulnessEval`) for decision making

### Tool Integration
Tools are bound to the LLM using `.bind_tools()` and executed via `ToolNode`:
- Search: Tavily API
- Research: Arxiv API
- RAG: Query vector store built from PDFs in `data/`

## Development Tips

### Adding New Graphs
1. Create new file in `app/graphs/`
2. Define graph using `StateGraph` builder pattern
3. Register in `langgraph.json` under `graphs` section

### Modifying Tools
- Add new tools in `app/tools.py`
- Tools must follow LangChain tool interface
- Update `tools` list passed to `ToolNode`

### Testing Changes
- Use `test_served_graph.py` as a template for integration tests
- Server must be running (`uv run langgraph dev`) before testing

### Common Patterns
- State updates use `TypedDict` with reducer functions
- Conditional edges use `END` to terminate graph execution
- Tool calls trigger automatic routing to `ToolNode`