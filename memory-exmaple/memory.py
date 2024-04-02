import semantic_kernel as sk
import asyncio

from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding
)

from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

collection_id = "generic"


async def populate_memory(memory: SemanticTextMemory) -> None:
    # Add some documents to the semantic memory
    await memory.save_information(collection=collection_id, id="info1", text="Your budget for 2024 is $100,000")
    await memory.save_information(collection=collection_id, id="info2", text="Your savings from 2023 are $50,000")
    await memory.save_information(collection=collection_id, id="info3", text="Your investments are $80,000")


async def search_memory_examples(memory: SemanticTextMemory) -> None:
    questions = ["What is my budget for 2024?", "What are my savings from 2023?", "What are my investments?"]

    for question in questions:
        print(f"Question: {question}")
        result = await memory.search(collection_id, question)
        print(f"Answer: {result[0].text}\n")
        

async def main():
    kernel = sk.Kernel()
    chat_service_id = "chat"

    api_key, org_id = sk.openai_settings_from_dot_env()
    oai_chat_service = OpenAIChatCompletion(
        service_id=chat_service_id, ai_model_id="gpt-3.5-turbo", api_key=api_key, org_id=org_id
    )
    embedding_gen = OpenAITextEmbedding(ai_model_id="text-embedding-ada-002", api_key=api_key, org_id=org_id)
    kernel.add_service(oai_chat_service)
    kernel.add_service(embedding_gen)
    memory = SemanticTextMemory(storage=sk.memory.VolatileMemoryStore(), embeddings_generator=embedding_gen)
    kernel.import_plugin_from_object(TextMemoryPlugin(memory), "TextMemoryPlugin")

    await populate_memory(memory)
    await search_memory_examples(memory)


if __name__ == "__main__":
    asyncio.run(main())
    
