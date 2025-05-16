from openai import OpenAI


def main():
    client = OpenAI()

    parameter = {
        "model": "o4-mini",
        "reasoning": {
            "effort": "low",
            "summary": "auto"
        },
        "input": [{
            "role": "user",
            "content": "Create a Python function returns `true` if a given year is a leap year, and `false` otherwise."
        }],
        "max_output_tokens": 300,
    }

    response = client.responses.create(**parameter)

    print(response.status)
    print(response.incomplete_details)
    print(response.output_text)

    print(response.reasoning)


if __name__ == '__main__':
    main()
