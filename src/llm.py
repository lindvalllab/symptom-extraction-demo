import json
import time
import os
from openai import OpenAI
import openai
import traceback


#from src.schema import OutputSchema


def openai_chat_completion_response(
        client: openai.OpenAI,
        #client,
        system_message: str,
        user_message: str,
        output_schema:type,
        model: str = 'gpt-4.1-nano-2025-04-14',
        max_attempts: int = 5,
        **kwargs,
):
    def _completion(messages: list[dict[str, str]]):
        
        # Check if TOGETHER_API_KEY is set
        if os.getenv('TOGETHER_API_KEY'):
            # Use Together AI format
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "schema": output_schema.model_json_schema(),
                },
                **kwargs,
            )
            output = json.loads(completion.choices[0].message.content)
        else:
            # Use OpenAI format
            completion = client.responses.parse(
                model=model,
                input=messages,
                text_format=output_schema,
                #tool_choice=output_schema.tool_choice,
                **kwargs,
            )
            output = completion.output_text
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
            print(f"Error: {e}")
            print(f"Attempt {attempts} failed")
            traceback.print_exc()
            
            if attempts < max_attempts:
                sleep_time = 2 ** (attempts - 2)  # Exponential backoff formula
                print(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)

            else:
                print("Max attempts reached, handling failure...")
                return None  # Return None or an appropriate failure indicator
