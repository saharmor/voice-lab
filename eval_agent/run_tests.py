# inject interrupts
# Gemini 2.0 and control when it starts talking


# call gemini, stream audio back, and hold it back until interupttion passes
import asyncio
import threading
from gemini_connection import GeminiConnection

from dotenv import load_dotenv

load_dotenv()


def run_tests():
    """Run a simple voice conversation with Gemini"""
    # Set up basic config
    config = {
        "system_prompt": "You are a friendly Gemini 2.0 model. Respond verbally in a casual, helpful tone.",
        "voice": "Puck",
        "google_search": False,
        "allow_interruptions": True
    }
    
    # Create cleanup event for graceful shutdown
    cleanup_event = threading.Event()
    
    # Initialize Gemini connection
    gemini = GeminiConnection(config, cleanup_event)
    
    try:
        # Run the conversation
        asyncio.run(gemini.start())
    except KeyboardInterrupt:
        print("\nStopping conversation...")
    finally:
        # Clean up
        cleanup_event.set()


run_tests()
