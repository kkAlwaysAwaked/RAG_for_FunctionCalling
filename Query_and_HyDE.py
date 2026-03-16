# 实现了 Rewrite 和 HyDE 逻辑
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Configs
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions" # DeepSeek 的官方接口地址

# Query Rewrite
async def rewrite_query(chat_history : list, latest_query : str):
    system_prompt = """
    给定以下对话历史和用户最新提问，请将最新提问重写为一个独立、完整、无需上下文就能看懂的句子。
    如果不需要重写，直接输出原句。不要回答问题，只输出重写后的句子。
    """

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    user_content = f"历史对话：\n{history_str}\n\n最新提问：{latest_query}"

    # http请求头
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # http请求体
    payload = {
        "model" : "deepseek-chat",
        "messages" : [
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : user_content}
        ],
        "temperature" : 0.0
    }

    async with httpx.AsyncClient() as client:
        try :
            response = await client.post(
                DEEPSEEK_API_URL,
                headers = headers,
                json = payload,
                timeout = 10.0
            )
            response.raise_for_status()
            # 解析返回的json数据
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except httpx.HTTPError as e:
            print("网络请求失败！")

# HyDE
async def generate_hyde_document(query : str) -> str:
    system_prompt = """
    请写一段话回答用户的问题。你需要模仿官方技术文档的语气，即使你不确定，也要尽可能写出相关的结构和专业术语。
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "temperature": 0.7  # 0.7 允许模型有一定的“幻觉”和词汇扩展
    }

    async with httpx.AsyncClient() as client :
        try :
            response = await client.post(
                DEEPSEEK_API_URL,
                headers = headers,
                json = payload,
                timeout = 15.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except httpx.HTTPError as e:
            print(f"HyDE 生成失败: {e}")
            return ""   # 如果失败，返回空字符串，后续检索时直接退化为只搜原始 Query


async def generate_hyde_vector(messages: list):
    """
    从标准 messages 列表中提取历史对话和最新提问，并调用 Query Rewrite 和 HyDE。
    返回用于后续向量检索的关键文本。
    """
    if not messages:
        return None, None

    # 1. 提取最新提问和历史对话
    latest_query = messages[-1].get("content", "")
    chat_history = messages[:-1]

    # 2. 核心步骤一：执行 Query 改写
    rewritten_query = await rewrite_query(chat_history=chat_history, latest_query=latest_query)

    # 如果改写失败（比如网络波动），降级使用原始的 latest_query
    if not rewritten_query:
        rewritten_query = latest_query

    # 3. 核心步骤二：基于【改写后的独立问题】生成假设性文档 (HyDE)
    hyde_doc = await generate_hyde_document(query=rewritten_query)

    # 4. 返回结果供下游 Embedding 和检索使用
    # ***返回格式建议为一个字典，方便后续扩展
    return {
        "rewritten_query": rewritten_query, # 改写后的query
        "hyde_document": hyde_doc   # 生成的假设性文档
    }







