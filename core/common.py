import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
)
from openai import OpenAI


llm = ChatOpenAI(
    model_name="qwen-max-latest",
    # openai_api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    streaming=True,
)

system_message_template = ChatMessagePromptTemplate.from_template(
    template="你是一个{role}的专家，擅长回答{domain}领域的问题。", role="system"
)
human_message_template = ChatMessagePromptTemplate.from_template(
    template="用户问题：{question}", role="user"
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_message_template,
        human_message_template,
    ]
)


# vector embedding client
load_dotenv()
rag_client = OpenAI(api_key=os.getenv("api_key"), base_url=os.getenv("base_url"))