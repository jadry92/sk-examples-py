import asyncio
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.contents.chat_history import ChatHistory

def create_chat_funtion(kernel: sk.Kernel):

    # Prepare OpenAI service using credentials stored in the `.env` file
    api_key, org_id = sk.openai_settings_from_dot_env()
    service_id="aoi_chat_gpt"
    kernel.add_service(
        sk_oai.OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-3.5-turbo",
            api_key=api_key,
            org_id=org_id
        )
    )
    
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if it doesn't have the answer.

    {{$history}}
    User: {{$user_input}}
    ChatBot: """
    

    execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.7
    )

    prompt_template_config = sk.PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
                InputVariable(name="input", description="The user input", is_required=True),
                InputVariable(name="history", description="The conversation history", is_required=False)
            ],
        execution_settings=execution_settings,
    )

    return kernel.create_function_from_prompt(
        function_name="chat",
        plugin_name="chat_plugin",
        prompt_template_config=prompt_template_config
    )


async def chat(kernel: sk.Kernel, history: ChatHistory):
    question = input("user: ")
    if "exit" in question or "quit" in question or "done" in question:
        return "exit"
    history.add_user_message(question) 
    arguments = KernelArguments(user_input=question, history=history)
    response = await kernel.invoke(chat_function, arguments)
    print(f"ChatBot: {response}")
    history.add_assistant_message(str(response))


if __name__ == "__main__":
    
    pass
