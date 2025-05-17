"""
reasoning_example.py

This module demonstrates the usage of the OpenAI Responses API for reasoning tasks. 

Links:
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses/create
- Reasoning Best Practices: https://platform.openai.com/docs/guides/reasoning-best-practices
- Responses vs Chat Completions: https://platform.openai.com/docs/guides/responses-vs-chat-completions
"""
import os
from openai import OpenAI


def reasoning():
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Define the parameters for the reasoning request
    client_parameter = {
        "model": "o4-mini",
        "reasoning": {
            "effort": "low",
            "summary": "auto"
        },
        "input": [{
            "role": "user",
            "content": "Create a Python function that returns `true` if a given year is a leap year, and `false` otherwise."
        }],
        "max_output_tokens": 300,
    }

    try:
        # Make the API call
        response = client.responses.create(**client_parameter)

        # Validate and print the response
        if hasattr(response, 'status') and hasattr(response, 'output_text'):
            print(f"Response Status: {response.status}")
            print(f"Incomplete Details: {response.incomplete_details}")
            print(f"Output Text:\n{response.output_text}")
            print(f"Reasoning Details: {response.reasoning}")
        else:
            print("Unexpected response format. Please check the API documentation.")

    except Exception as e:
        # Handle any errors that occur during the API call
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    reasoning()
