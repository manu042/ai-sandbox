"""
websearch_stream.py

This script demonstrates the use of the Anthropic API to perform a web search. 

Links:
    https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
    https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/web_search_stream.py
"""
import os
import logging
from typing import Dict, List, Optional
import anthropic
from anthropic.types import MessageStopEvent, MessageStartEvent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicAPI:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.history: List[Dict[str, str]] = []
        self.citation_buffer: Optional[str] = None

    def create_client_parameters(self) -> Dict:
        """Define the parameters for the Claude API.

        Returns:
            dict: A dictionary containing client parameters such as model, tokens, temperature, and tools.
        """
        user_location = {
                "type": "approximate",
                "country": "GB",
                "city": "London",
                "region": "London",
        }

        web_tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
            "user_location": user_location,
        }

        client_parameters = {
            "model": "claude-3-7-sonnet-latest",
            "max_tokens": 500,
            "top_p": 1.0,
            "temperature": 1.0,
            "system": "You are a helpful AI assistant. Your answers shoud be short and precise.",
            "tools": [web_tool],
        }

        return client_parameters

    def process_event(self, event: MessageStartEvent) -> None:
        """Process events received from the Anthropic API.

        Args:
            event (MessageStartEvent): The event to process, which can include content blocks, citations, or messages.
        """
        if event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(f"{event.delta.text}", end="")

        elif event.type == "citation":
            self.citation_buffer = f"[{event.citation.title}]({event.citation.url})"

        elif event.type == "content_block_start":
            print(f"Content Block Start - Index: {event.index} - Type: {event.content_block.type}")

        elif event.type == "content_block_stop":
            if self.citation_buffer:
                print(self.citation_buffer)
                self.citation_buffer = ""

            print(f"Content Block Stop - Index: {event.index} - Type: {event.content_block.type}")

        elif event.type in ("message_start"):
            print(f"{event.type}")

        elif isinstance(event, MessageStopEvent):
            # Add assistant response to history
            self.history.append({
                "role": "assistant",
                "content": event.message.content
            })

            print(f"Message ID: {event.message.id}")
            print(f"Stop reason: {event.message.stop_reason}")

            print("Usage:")
            print(f"\t{event.message.usage.input_tokens}")
            print(f"\t{event.message.usage.output_tokens}")
            print(f"\t{event.message.usage.server_tool_use}")

            print(f"Content length: {len(event.message.content)}")
            for content in event.message.content:

                if content.type == "web_search_tool_result":
                    for web_result in content.content:
                        print(f"[{web_result.title}]({web_result.url})")

        # Ignore
        elif event.type in ("message_delta", "text"):
            pass

        else:
            print(event.type)
        
        print()

    def send_message(self, message: str) -> None:
        """Send a message to the Anthropic API and process the response.

        Args:
            message (str): The user message to send.
        """
        # Add user message to history
        self.history.append({
            "role": "user",
            "content": message
        })

        parameter = self.create_client_parameters()

        try:
            with self.client.messages.stream(messages=self.history, **parameter) as stream:
                for event in stream:
                    self.process_event(event)
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
        except anthropic.APIConnectionError as e:
            logger.error(f"Connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

def main():
    ant = AnthropicAPI()
    ant.send_message("What was a positive news story from today?")


if __name__ == '__main__':
    main()
