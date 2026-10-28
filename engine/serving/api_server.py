from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16_384)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)


class GenerateResponse(BaseModel):
    request_id: str
    text: str


@dataclass
class PendingRequest:
    request_id: str
    payload: GenerateRequest
    future: asyncio.Future[str]


class ContinuousBatcher:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None
        self.model = None
        self.tokenizer = None

    def _load_model(self) -> None:
        model_id = os.getenv("MODEL_ID")
        if not model_id:
            raise RuntimeError("MODEL_ID must be configured before starting the generation service")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def start(self) -> None:
        await asyncio.to_thread(self._load_model)
        self.worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)

    async def submit(self, payload: GenerateRequest) -> PendingRequest:
        loop = asyncio.get_running_loop()
        request = PendingRequest(str(uuid.uuid4()), payload, loop.create_future())
        await self.queue.put(request)
        return request

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            await asyncio.sleep(0.002)
            while not self.queue.empty() and len(batch) < 32:
                batch.append(self.queue.get_nowait())
            prompts = [request.payload.prompt for request in batch]
            tokenizer = self.tokenizer
            model = self.model
            if tokenizer is None or model is None:
                raise RuntimeError("generation model failed to initialize")
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            max_new_tokens = max(request.payload.max_new_tokens for request in batch)
            generated = await asyncio.to_thread(
                model.generate,
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(request.payload.temperature for request in batch),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[1]
            for index, request in enumerate(batch):
                text = tokenizer.decode(generated[index, prompt_length:], skip_special_tokens=True)
                if not request.future.done():
                    request.future.set_result(text)


app = FastAPI(title="DistributedLLM-Engine", version="0.1.0")
batcher = ContinuousBatcher()


@app.on_event("startup")
async def startup() -> None:
    await batcher.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await batcher.stop()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/completions", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    request = await batcher.submit(payload)
    try:
        text = await asyncio.wait_for(request.future, timeout=120)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="generation timeout") from exc
    return GenerateResponse(request_id=request.request_id, text=text)


@app.post("/v1/completions/stream")
async def stream(payload: GenerateRequest) -> EventSourceResponse:
    request = await batcher.submit(payload)

    async def events() -> AsyncIterator[dict[str, str]]:
        yield {"event": "request", "data": request.request_id}
        try:
            result = await asyncio.wait_for(request.future, timeout=120)
            for token in result.split():
                yield {"event": "token", "data": token + " "}
                await asyncio.sleep(0)
            yield {"event": "done", "data": request.request_id}
        except asyncio.TimeoutError:
            yield {"event": "error", "data": "generation timeout"}

    return EventSourceResponse(events())
