
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.planners.basic_planner import BasicPlanner

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
import os
import asyncio

async def main():

    kernel = sk.Kernel()
    service_id = "shakespeare"
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_service(
        sk_oai.OpenAIChatCompletion(
            service_id=service_id, ai_model_id="gpt-3.5-turbo-1106", api_key=api_key, org_id=org_id
        ),
    )

    plugins_dir = os.path.split(os.path.abspath(__file__))[0]

    kernel.import_plugin_from_prompt_directory(plugins_dir, "SummarizePlugin")
    kernel.import_plugin_from_prompt_directory(plugins_dir, "WriterPlugin")
    kernel.import_plugin_from_object(TextPlugin(), "TextPlugin")
    
    
    for plugin in kernel.plugins:
        for function in plugin.functions.values():
            print(f"Plugin: {plugin.name}, Function: {function.name}")

    print(kernel.plugins["SummarizePlugin"].functions["Notegen"].prompt_template)
    
    # ask = """
    # Tomorrow is Valentine's day. I need to come up with a few short poems.
    # She likes Shakespeare so write using his style. She speaks Spanish so write it in Spanish.
    # Convert the text to uppercase."""
    #
    # planner = BasicPlanner(service_id)
    # basic_plan = await planner.create_plan(goal=ask, kernel=kernel)
    # print(basic_plan.generated_plan)
    # result = await planner.execute_plan(basic_plan, kernel)
    # print(result)
    #


if __name__ == "__main__":
    asyncio.run(main())
