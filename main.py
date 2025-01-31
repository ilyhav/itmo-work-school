import os
import time
import re
from typing import List, Optional
import xml.etree.ElementTree as ET

import openai
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl

from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger

app = FastAPI()
logger = None

OPENAI_API_KEY = os.environ.get("OPENAI_TOKEN")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Настраиваем Yandex XML Search API
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "")
YANDEX_USER = os.environ.get("YANDEX_USER", "")

def extract_choices(query_text: str) -> List[str]:
    """
    Ищет варианты ответов вида "1. ...", "2. ...", и т.д.
    """
    pattern = re.compile(r'^(?:\d{1,2}\.)\s?.+', re.MULTILINE)
    choices = pattern.findall(query_text.strip())
    return [c.strip() for c in choices]

def call_openai_for_answer(question: str, choices: List[str]) -> (Optional[int], str):
    if choices:
        system_prompt = "Ты — эксперт по Университету ИТМО. Определи правильный вариант ответа на заданный вопрос."
        user_prompt = (
            "Ниже приведены вопрос и варианты ответов. Определи, какой номер ответа является правильным.\n\n"
            f"Вопрос: {question}\n\n"
            "Варианты:\n" + "\n".join(choices) + "\n"
            "Ответь только числом (например, 1,2,3...), без пояснений."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            text = response.choices[0].message["content"].strip()
            match = re.search(r'\b(\d{1,2})\b', text)
            if match:
                chosen = int(match.group(1))
                reasoning = (
                    f"Ответ сгенерирован моделью GPT-3.5-turbo. Определение ответа: модель вернула вариант {chosen}."
                )
                return chosen, reasoning
            else:
                return None, "OpenAI не смогла определить вариант ответа."
        except Exception as e:
            return None, f"OpenAI вызвал исключение: {str(e)}"
    else:
        system_prompt = "Ты — эксперт по Университету ИТМО, предоставляющий исчерпывающую информацию."
        user_prompt = (
            f"Ответь на следующий вопрос, предоставив подробное объяснение и, при возможности, укажи источники:\n\n{question}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            text = response.choices[0].message["content"].strip()
            reasoning = f"Ответ сгенерирован моделью GPT-3.5-turbo. {text}"
            return None, reasoning
        except Exception as e:
            return None, f"OpenAI вызвал исключение: {str(e)}"

async def get_sources(query: str) -> List[str]:
    if not YANDEX_API_KEY:
        return ["https://itmo.ru/ru/", "https://news.itmo.ru/"]

    is_news = any(keyword in query.lower() for keyword in ["новости", "рейтинг", "топ"])
    search_query = query + (" новости ИТМО" if is_news else " ИТМО")

    url = "https://yandex.com/search/xml"
    params = {
        "query": search_query,
        "l10n": "ru",
        "sortby": "rlv",
        "groupby": "groups-on-page=3",
    }
    if YANDEX_USER:
        params["user"] = YANDEX_USER
    params["key"] = YANDEX_API_KEY

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()

            tree = ET.fromstring(resp.text)
            results = []
            for doc in tree.findall(".//doc"):
                url_element = doc.find("url")
                if url_element is not None:
                    results.append(url_element.text)
            return results[:3]
        except Exception as e:
            if logger:
                await logger.error(f"Ошибка при поиске источников через Yandex: {str(e)}")
            return ["https://itmo.ru/ru/", "https://news.itmo.ru/"]

@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    if logger:
        await logger.info(
            f"Incoming request: {request.method} {request.url}\n"
            f"Request body: {body.decode()}"
        )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    if logger:
        await logger.info(
            f"Request completed: {request.method} {request.url}\n"
            f"Status: {response.status_code}\n"
            f"Response body: {response_body.decode()}\n"
            f"Duration: {process_time:.3f}s"
        )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        if logger:
            await logger.info(f"Processing prediction request with id: {body.id}")

        choices = extract_choices(body.query)

        answer, reasoning = call_openai_for_answer(body.query, choices)

        sources_raw = await get_sources(body.query)

        validated_sources = []
        for s in sources_raw:
            try:
                validated_sources.append(HttpUrl(s))
            except Exception:
                continue

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=reasoning,
            sources=validated_sources,
        )
        if logger:
            await logger.info(f"Successfully processed request {body.id}")
        return response

    except ValueError as e:
        error_msg = str(e)
        if logger:
            await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        if logger:
            await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
