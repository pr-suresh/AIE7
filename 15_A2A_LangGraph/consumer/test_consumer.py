"""Test script for the consumer agent.

This script demonstrates how the consumer agent works by sending various
test queries and showing the responses.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

try:
    from .consumer_agent import ConsumerAgent
except ImportError:
    from consumer_agent import ConsumerAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_consumer_agent():
    """Test the consumer agent with various queries."""
    print("🧪 Testing Consumer Agent")
    print("="*50)
    
    # Initialize consumer agent
    consumer = ConsumerAgent()
    
    try:
        # Test connection
        print("\n1. Testing connection to provider...")
        success = await consumer.initialize_a2a_client()
        
        if not success:
            print("❌ Failed to connect to provider. Make sure it's running on http://localhost:10000")
            return
        
        print("✅ Connected successfully!")
        print(f"Provider: {consumer.agent_card.name}")
        
        # Test queries
        test_queries = [
            "Hello, how are you?",  # Simple greeting (should be handled directly)
            "What is 2+2?",  # Simple math (should be handled directly)
            "What are the latest developments in artificial intelligence?",  # Web search needed
            "Find me recent papers on transformer architectures",  # Academic search needed
            "What do the policy documents say about student loans?",  # RAG search needed
        ]
        
        print(f"\n2. Testing {len(test_queries)} different types of queries...")
        print("="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 40)
            
            try:
                response = await consumer.run(query, thread_id=f"test_{i}")
                print(f"Response: {response}")
                
                # Add a small delay between requests
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Error on query {i}: {e}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await consumer.cleanup()


async def test_conversation_memory():
    """Test conversation memory and multi-turn interactions."""
    print("\n🧠 Testing Conversation Memory")
    print("="*40)
    
    consumer = ConsumerAgent()
    
    try:
        await consumer.initialize_a2a_client()
        
        # Multi-turn conversation
        conversation = [
            "Find me information about machine learning",
            "Can you summarize the key points?",
            "What are the practical applications?"
        ]
        
        thread_id = "memory_test"
        
        for i, message in enumerate(conversation, 1):
            print(f"\nTurn {i}: {message}")
            response = await consumer.run(message, thread_id=thread_id)
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            
            await asyncio.sleep(1)
    
    finally:
        await consumer.cleanup()


if __name__ == "__main__":
    async def run_all_tests():
        """Run all tests."""
        await test_consumer_agent()
        await test_conversation_memory()
    
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted!")
