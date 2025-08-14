"""Demo script for the Consumer Agent.

This script demonstrates the A2A communication between the consumer and provider agents.
Run this after starting the provider agent with: uv run python -m app
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from consumer.consumer_agent import ConsumerAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_consumer_agent():
    """Demonstrate the consumer agent functionality."""
    print("🎭 Consumer Agent Demo")
    print("="*60)
    print("This demo shows how the Consumer Agent communicates with")
    print("the Provider Agent via the A2A protocol.")
    print("="*60)
    
    # Check if environment is set up
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key and try again.")
        return
    
    # Initialize consumer agent
    print("\n🔧 Initializing Consumer Agent...")
    consumer = ConsumerAgent()
    
    try:
        # Test connection to provider
        print("\n📡 Connecting to Provider Agent...")
        success = await consumer.initialize_a2a_client()
        
        if not success:
            print("❌ Failed to connect to Provider Agent!")
            print("\nTo fix this:")
            print("1. Make sure the provider agent is running:")
            print("   cd /Users/psuresh/Documents/2025AIEngineer/AIE7/15_A2A_LangGraph")
            print("   uv run python -m app")
            print("2. Wait for it to start on http://localhost:10000")
            print("3. Run this demo again")
            return
        
        print("✅ Successfully connected to Provider Agent!")
        print(f"\nProvider Details:")
        print(f"  Name: {consumer.agent_card.name}")
        print(f"  Description: {consumer.agent_card.description}")
        print(f"  Available Skills:")
        for skill in consumer.agent_card.skills:
            print(f"    • {skill.name}: {skill.description}")
        
        # Demo queries
        demo_queries = [
            {
                "query": "Hello! What's 2 + 2?",
                "description": "Simple Math (should be handled directly by consumer)",
                "expected": "Direct response without A2A call"
            },
            {
                "query": "What are the latest AI research trends?",
                "description": "Current Information (should be forwarded to provider for web search)",
                "expected": "Response from provider using web search tools"
            },
            {
                "query": "Find me recent papers on large language models",
                "description": "Academic Search (should use provider's arXiv search)",
                "expected": "Response from provider using arXiv search"
            }
        ]
        
        print(f"\n🎯 Running {len(demo_queries)} Demo Queries")
        print("="*60)
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n🔍 Demo {i}: {demo['description']}")
            print(f"Query: \"{demo['query']}\"")
            print(f"Expected: {demo['expected']}")
            print("-" * 50)
            
            try:
                # Run the query
                print("🤔 Processing...")
                response = await consumer.run(demo['query'], thread_id=f"demo_{i}")
                
                # Display response
                print(f"✅ Response:")
                print(f"{response}")
                
                # Add delay between queries
                if i < len(demo_queries):
                    print("\n⏳ Waiting 2 seconds before next query...")
                    await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Error processing query {i}: {e}")
                logger.exception(f"Error on demo query {i}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nWhat you've seen:")
        print("• Consumer agent intelligently routing queries")
        print("• Direct responses for simple queries")
        print("• A2A protocol communication for complex queries")
        print("• Provider agent using its tools (web search, arXiv, RAG)")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.exception("Demo failed")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await consumer.cleanup()
        print("✅ Cleanup complete!")


if __name__ == "__main__":
    try:
        print("Starting Consumer Agent Demo...")
        print("Press Ctrl+C to stop at any time.\n")
        asyncio.run(demo_consumer_agent())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)

