"""
websearch_response.py

This script demonstrates the use of the OpenAI Responses API to perform a web search and process the results. 

Links:
    https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses
    https://platform.openai.com/docs/api-reference/responses/create
"""
import logging
from openai import OpenAI
from openai.types.responses import Response, ResponseFunctionWebSearch, ResponseOutputMessage, ResponseOutputText, WebSearchTool


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_client_parameters():
    """Define the parameters for the reasoning request."""
    return {
        "model": "gpt-4o-mini",
        "max_output_tokens": 1000,
        "top_p": 1.0,
        "temperature": 1.0,
        "store": False,
        "instructions": "You are a helpful assistant",
        "tools": [
            {
                "type": "web_search_preview",
                "search_context_size": "low",
                "user_location": {
                    "type": "approximate",
                    "country": "GB",
                    "city": "London",
                    "region": "London",
                },
            }
        ],
        "input": "What was a positive news story from today?",
    }


def process_response(response: Response):
    """Process and log the response details."""
    logging.info("Response ID: %s", response.id)

    if response.error:
        logging.error("Error: %s", response.error)
        return
    else:
        logging.info("Response status: %s", response.status)
        logging.info(f"Incomplete Details: {response.incomplete_details}")

    logging.info("Model: %s", response.model)
    logging.info("Instructions: %s", response.instructions)
    logging.info("Temperature: %s", response.temperature)
    logging.info("Top P: %s", response.top_p)
    logging.info("Max Output Tokens: %s", response.max_output_tokens)
    logging.info("Service Tier: %s", response.service_tier)

    logging.info("Store: %s", response.store)
    logging.info("Truncation: %s", response.truncation)
    logging.info("Previous response id: %s", response.previous_response_id)

    # Log token usage
    if response.usage:
        logging.info(
            "Token Usage - Total: %s, Input: %s, Output: %s",
            response.usage.total_tokens,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        logging.info("Input token details: %s", response.usage.input_tokens_details)
        logging.info("Output token details: %s", response.usage.output_tokens_details)

    # Log tool information
    if response.tools:
        logging.info("Parallel Tool calls: %s", response.parallel_tool_calls)
        logging.info("Tool choice: %s", response.tool_choice)

        for tool in response.tools:
            if isinstance(tool, WebSearchTool):
                logging.info("Tool Type: %s", tool.type)
                logging.info("Search Context Size: %s", tool.search_context_size)
                logging.info(
                    "User Location: %s, %s, %s, %s",
                    tool.user_location.city,
                    tool.user_location.region,
                    tool.user_location.country,
                    tool.user_location.timezone,
                )

    # Log output details
    if response.output:
        for output in response.output:
            if isinstance(output, ResponseFunctionWebSearch):
                logging.info("Web Search Output - Status: %s, Type: %s", output.status, output.type)

            elif isinstance(output, ResponseOutputMessage):
                logging.info("Message Output - Role: %s, Status: %s, Type: %s", output.role, output.status, output.type)

                for content in output.content:
                    if isinstance(content, ResponseOutputText):
                        logging.info("Text Content: %s", content.text)

                        for annotation in content.annotations:
                            logging.info("Annotation - Type: %s, Title: %s, URL: %s", annotation.type, annotation.title, annotation.url)


def websearch():
    """Perform a web search using the OpenAI Responses API."""
    client = OpenAI()
    client_parameters = create_client_parameters()

    try:
        response: Response = client.responses.create(**client_parameters)
        process_response(response)
    except Exception as e:
        logging.error("An error occurred while making the API request: %s", e)


if __name__ == '__main__':
    websearch()
