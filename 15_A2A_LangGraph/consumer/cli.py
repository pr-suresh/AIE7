"""CLI interface for the consumer agent.

This provides a simple command-line interface to interact with the consumer agent
that communicates with the provider via A2A protocol.
"""

import asyncio
import logging
import sys
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


async def main():
    """Main CLI function."""
    print("🤖 Consumer Agent CLI")
    print("This agent can communicate with the provider agent via A2A protocol.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Initialize consumer agent
    consumer = ConsumerAgent()
    
    try:
        # Test connection to provider
        print("Initializing connection to provider agent...")
        success = await consumer.initialize_a2a_client()
        
        if not success:
            print("❌ Failed to connect to provider agent.")
            print("Please ensure the provider agent is running on http://localhost:10000")
            print("Run: uv run python -m app")
            return
        
        print("✅ Successfully connected to provider agent!")
        print(f"Provider: {consumer.agent_card.name}")
        print(f"Description: {consumer.agent_card.description}")
        print("Available skills:")
        for skill in consumer.agent_card.skills:
            print(f"  - {skill.name}: {skill.description}")
        print("\n" + "="*60 + "\n")
        
        # Interactive loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                print("🤔 Processing...")
                response = await consumer.run(user_input)
                print(f"Agent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    finally:
        # Cleanup
        await consumer.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
