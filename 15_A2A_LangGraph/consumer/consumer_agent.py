"""Consumer Agent that uses the A2A protocol to communicate with the provider agent.

This consumer agent demonstrates how to build a LangGraph agent that makes API calls
to another agent through the A2A (Agent-to-Agent) protocol.
"""

import logging
import os
from typing import Dict, Any, List
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from uuid import uuid4

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsumerState(BaseModel):
    """State for the consumer agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_user_input: str = ""
    provider_response: str = ""
    task_complete: bool = False


class ConsumerAgent:
    """A consumer agent that uses A2A protocol to communicate with a provider agent."""
    
    def __init__(self, provider_base_url: str = "http://localhost:10000"):
        """Initialize the consumer agent.
        
        Args:
            provider_base_url: The base URL of the provider agent server
        """
        self.provider_base_url = provider_base_url
        self.httpx_client = None
        self.a2a_client = None
        self.agent_card = None
        
        # Initialize the LLM for the consumer agent
        self.llm = ChatOpenAI(
            model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
            temperature=0.1,
        )
        
        # Memory for conversation state
        self.memory = MemorySaver()
        
        # Build the consumer graph
        self.graph = self._build_consumer_graph()
    
    async def initialize_a2a_client(self):
        """Initialize the A2A client to connect to the provider."""
        try:
            self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            
            # Initialize A2A card resolver
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.provider_base_url,
            )
            
            # Fetch the provider's agent card
            logger.info(f"Fetching agent card from {self.provider_base_url}")
            self.agent_card = await resolver.get_agent_card()
            logger.info(f"Successfully connected to provider: {self.agent_card.name}")
            logger.info(f"Provider capabilities: {self.agent_card.capabilities}")
            logger.info(f"Provider skills: {[skill.name for skill in self.agent_card.skills]}")
            
            # Initialize A2A client
            self.a2a_client = A2AClient(
                httpx_client=self.httpx_client,
                agent_card=self.agent_card
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize A2A client: {e}")
            return False
    
    async def send_message_to_provider(self, message: str) -> str:
        """Send a message to the provider agent via A2A protocol.
        
        Args:
            message: The message to send to the provider
            
        Returns:
            The response from the provider agent
        """
        if not self.a2a_client:
            raise RuntimeError("A2A client not initialized. Call initialize_a2a_client() first.")
        
        try:
            # Prepare the message payload
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': message}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            
            # Create the request
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # Send the message and get response
            logger.info(f"Sending message to provider: {message}")
            response = await self.a2a_client.send_message(request)
            
            # Extract the response content
            if response.root and response.root.result and response.root.result.artifacts:
                # Get the first artifact content
                artifact = response.root.result.artifacts[0]
                if artifact.parts and len(artifact.parts) > 0:
                    content = artifact.parts[0].root.text
                    logger.info(f"Received response from provider: {content[:100]}...")
                    return content
            
            return "No response received from provider"
            
        except Exception as e:
            logger.error(f"Error sending message to provider: {e}")
            return f"Error communicating with provider: {str(e)}"
    
    def _build_consumer_graph(self) -> StateGraph:
        """Build the LangGraph consumer graph."""
        
        async def consumer_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Main consumer node that processes user input and decides next action."""
            messages = state["messages"]
            last_message = messages[-1]
            
            if isinstance(last_message, HumanMessage):
                user_input = last_message.content
                
                # Use LLM to analyze the request and determine if we need the provider
                analysis_prompt = f"""
                Analyze this user request: "{user_input}"
                
                This consumer agent can:
                1. Answer simple questions directly
                2. Forward complex requests to a provider agent that has web search, academic paper search, and document retrieval capabilities
                
                Should this request be forwarded to the provider agent? 
                Respond with either:
                - "FORWARD: <reason>" if it should be forwarded
                - "DIRECT: <simple_answer>" if you can answer directly
                
                Provider agent skills:
                - Web Search Tool: Search the web for current information
                - Academic Paper Search: Search for academic papers on arXiv  
                - Document Retrieval: Search through loaded documents for specific information
                """
                
                analysis_response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
                analysis = analysis_response.content
                
                if analysis.startswith("FORWARD:"):
                    # Forward to provider
                    return {
                        "messages": [AIMessage(content="I'll help you with that. Let me consult the provider agent...")],
                        "original_user_input": user_input,
                        "provider_response": "",
                        "task_complete": False
                    }
                else:
                    # Answer directly
                    direct_answer = analysis.replace("DIRECT:", "").strip()
                    return {
                        "messages": [AIMessage(content=direct_answer)],
                        "original_user_input": user_input,
                        "provider_response": "",
                        "task_complete": True
                    }
            
            return {"messages": messages, "task_complete": True}
        
        async def provider_communication_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Node that communicates with the provider agent via A2A."""
            # Get the original user input that was preserved
            user_input = state.get("original_user_input", "")
            
            if not user_input:
                return {
                    "messages": [AIMessage(content="I couldn't find your original request.")],
                    "task_complete": True
                }
            
            # Send to provider
            provider_response = await self.send_message_to_provider(user_input)
            
            return {
                "messages": [AIMessage(content=provider_response)],
                "provider_response": provider_response,
                "task_complete": True
            }
        
        def route_decision(state: Dict[str, Any]) -> str:
            """Decide whether to route to provider or end."""
            if state.get("task_complete", False):
                return END
            else:
                return "provider_communication"
        
        # Build the graph
        graph = StateGraph(dict)
        
        # Add nodes
        graph.add_node("consumer", consumer_node)
        graph.add_node("provider_communication", provider_communication_node)
        
        # Set entry point
        graph.set_entry_point("consumer")
        
        # Add edges
        graph.add_conditional_edges(
            "consumer",
            route_decision,
            {
                "provider_communication": "provider_communication",
                END: END
            }
        )
        
        graph.add_edge("provider_communication", END)
        
        return graph.compile(checkpointer=self.memory)
    
    async def run(self, user_input: str, thread_id: str = "default") -> str:
        """Run the consumer agent with user input.
        
        Args:
            user_input: The user's input message
            thread_id: Thread ID for conversation memory
            
        Returns:
            The agent's response
        """
        # Initialize A2A client if not already done
        if not self.a2a_client:
            success = await self.initialize_a2a_client()
            if not success:
                return "Failed to connect to provider agent. Please check if the provider is running."
        
        # Prepare input
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "original_user_input": "",
            "provider_response": "",
            "task_complete": False
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the graph
        try:
            result = await self.graph.ainvoke(inputs, config)
            
            # Extract the final response
            if result and "messages" in result:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content
            
            return "I encountered an issue processing your request."
            
        except Exception as e:
            logger.error(f"Error running consumer agent: {e}")
            return f"An error occurred: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
