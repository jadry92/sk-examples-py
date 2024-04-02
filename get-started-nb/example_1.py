
import asyncio
import semantic_kernel as sk
import os
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

def create_joke_function(kernel, plugin_directory: str):

    api_key, org_id = sk.openai_settings_from_dot_env()
    service_id = "default"
    kernel.add_service(
        OpenAIChatCompletion(service_id=service_id, ai_model_id="gpt-3.5-turbo-1106", api_key=api_key, org_id=org_id),
    )

    plugin = kernel.import_plugin_from_prompt_directory(plugin_directory, "FunPlugin")

    return plugin["Joke"]

async def main(kernel,joke_function):
    joke = await kernel.invoke(joke_function, sk.KernelArguments(input="dad joke", style="super silly"))
    print(joke)

if __name__ == "__main__":
    script_full_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_full_path)[0]
    kernel = sk.Kernel()
    joke_function = create_joke_function(kernel,script_dir)
    asyncio.run(main(kernel, joke_function))
