import json
import time
import openai

from src.schema import OutputSchema


def openai_chat_completion_response(
        client: openai.OpenAI,
        system_message: str,
        user_message: str,
        output_schema: OutputSchema,
        model: str = 'gpt-3.5-turbo-1106',
        max_attempts: int = 5,
        **kwargs,
):
    def _completion(messages: list[dict[str, str]]):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[output_schema.tool],
            tool_choice=output_schema.tool_choice,
            **kwargs,
        )
        output = completion.choices[0].message.tool_calls[0].function.arguments
        output = json.loads(output)

        return output

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message},
    ]

    attempts = 0
    while attempts < max_attempts:
        try:
            attempts += 1
            result = _completion(messages)

            # attempt to parse
            output_schema(**result)
            return result
        except Exception as e:
            print(f"Attempt {attempts} failed")
            if attempts < max_attempts:
                sleep_time = 2 ** (attempts - 2)  # Exponential backoff formula
                print(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)

            else:
                print("Max attempts reached, handling failure...")
                return None  # Return None or an appropriate failure indicator
