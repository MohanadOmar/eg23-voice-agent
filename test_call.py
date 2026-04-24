"""
EG23 Voice Agent — Full Conversation Version
Stack: Twilio + Deepgram (STT) + GPT-4o mini + OpenAI TTS
Dodo can now listen, think, and respond.
"""

import os
import asyncio
import base64
import json
import struct

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.rest import Client
from openai import AsyncOpenAI
import websockets
import uvicorn
import httpx
from dotenv import load_dotenv

load_dotenv()

# ─── audioop compatibility for Python 3.13 ───
try:
    import audioop
except ImportError:
    class audioop:
        @staticmethod
        def ratecv(data, width, nchannels, inrate, outrate, state,
                   weightA=1, weightB=0):
            n_in  = len(data) // (width * nchannels)
            n_out = int(n_in * outrate / inrate)
            result = bytearray()
            for i in range(n_out):
                src = min(int(i * inrate / outrate), n_in - 1)
                for c in range(nchannels):
                    offset = (src * nchannels + c) * width
                    result.extend(data[offset:offset + width])
            return bytes(result), None

        @staticmethod
        def lin2ulaw(data, width):
            fmt    = '<h' if width == 2 else '<b'
            n      = len(data) // width
            result = []
            for i in range(n):
                s = struct.unpack_from(fmt, data, i * width)[0]
                if width == 1:
                    s = s * 256
                s    = max(-32768, min(32767, s))
                sign = 0x80 if s < 0 else 0
                s    = abs(s) + 132
                exp  = 7
                for e in range(7, 0, -1):
                    if s >= (1 << (e + 3)):
                        exp = e
                        break
                mantissa = (s >> (exp + 3)) & 0x0F
                ulaw     = (~(sign | (exp << 4) | mantissa)) & 0xFF
                result.append(ulaw)
            return bytes(result)


# ─── CONFIG ───
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY    = os.getenv("DEEPGRAM_API_KEY")
SERVER_URL          = os.getenv("SERVER_URL", "").rstrip("/")
CALL_TO_NUMBER      = os.getenv("CALL_TO_NUMBER")
N8N_WEBHOOK_URL     = os.getenv("N8N_WEBHOOK_URL", "")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()


# ─── DODO SYSTEM PROMPT ───
DODO_PROMPT = """You are Dodo — EG23's AI sales assistant calling a lead who requested a free trial on eg23.net.

EG23 is an AI agency that builds AI agents, intelligent workflows, and automations for small businesses. Slogan: "We test more than we develop."

YOUR GOAL
1. Confirm you're speaking to the right person
2. Quickly understand their business and what they want to automate
3. Answer short questions if asked
4. Book a 15-minute setup call with the EG23 team

TONE
- Warm, direct, confident — like a real assistant, not a bot
- Short sentences. Natural speech. Never sound scripted
- Keep responses under 2-3 sentences unless answering a specific question
- Never say "I'm just an AI" or "I'm a chatbot"

RULES
- If they're busy, offer to call back
- If they're not interested, thank them briefly and end the call
- If asked about pricing, say: "It depends on your setup — the team will walk you through it on the call, but most businesses start under a hundred dollars for setup"
- Never make up features, timelines, or guarantees
- When wrapping up, confirm the call outcome clearly

FIRST MESSAGE (this is already spoken — don't repeat it):
"Hey, this is Dodo from EG23. You requested a free trial on our site, so I wanted to reach out quickly to confirm a couple of details. Do you have two minutes?"

Keep the conversation flowing naturally from there."""


# ─── HEALTH CHECK ───
@app.get("/")
async def health():
    return {"status": "EG23 Voice Agent running", "version": "full-conversation"}


# ─── TWILIO ANSWER WEBHOOK ───
@app.post("/answer")
async def answer(request: Request):
    ws_url = SERVER_URL.replace("https://", "wss://").replace("http://", "ws://")
    twiml  = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}/stream"/>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# ─── WEBSOCKET — FULL CONVERSATION LOOP ───
@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[WS] Twilio connected")

    stream_sid           = None
    conversation         = [{"role": "system", "content": DODO_PROMPT}]
    transcript_log       = []
    call_outcome         = "unknown"
    is_speaking          = False
    has_spoken_opening   = False

    # Deepgram WebSocket connection
    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2&encoding=mulaw&sample_rate=8000"
        "&channels=1&punctuate=true&interim_results=true"
        "&endpointing=500&utterance_end_ms=1200"
    )
    dg_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    try:
        async with websockets.connect(dg_url, additional_headers=dg_headers) as dg_ws:
            print("[DEEPGRAM] Connected")

            # ─── TWILIO → DEEPGRAM ───
            async def twilio_to_deepgram():
                nonlocal stream_sid, has_spoken_opening
                try:
                    async for raw in ws.iter_text():
                        msg   = json.loads(raw)
                        event = msg.get("event")

                        if event == "start":
                            stream_sid = msg["start"]["streamSid"]
                            print(f"[STREAM] Started: {stream_sid}")

                            # Speak opening message
                            if not has_spoken_opening:
                                has_spoken_opening = True
                                opening = (
                                    "Hey, this is Dodo from EG23. "
                                    "You requested a free trial on our site, so I wanted to reach out "
                                    "to confirm a couple of details. Do you have two minutes?"
                                )
                                print(f"[DODO] {opening}")
                                conversation.append({"role": "assistant", "content": opening})
                                transcript_log.append(f"Dodo: {opening}")
                                await speak(opening, stream_sid, ws)

                        elif event == "media":
                            audio_b64 = msg["media"]["payload"]
                            audio_bytes = base64.b64decode(audio_b64)
                            await dg_ws.send(audio_bytes)

                        elif event == "stop":
                            print("[STREAM] Stopped")
                            break

                except WebSocketDisconnect:
                    print("[WS] Twilio disconnected")
                except Exception as e:
                    print(f"[TWILIO→DG] Error: {e}")

            # ─── DEEPGRAM → GPT → TTS ───
            async def deepgram_to_response():
                nonlocal is_speaking, call_outcome
                buffer = ""

                try:
                    async for raw in dg_ws:
                        dg_msg = json.loads(raw)

                        # Handle utterance end (user finished speaking)
                        if dg_msg.get("type") == "UtteranceEnd":
                            if buffer.strip() and not is_speaking:
                                user_text = buffer.strip()
                                buffer = ""

                                print(f"[USER] {user_text}")
                                transcript_log.append(f"Lead: {user_text}")

                                # Detect outcome signals
                                lower = user_text.lower()
                                if any(w in lower for w in ["not interested", "no thanks", "don't want", "stop calling"]):
                                    call_outcome = "not_interested"
                                elif any(w in lower for w in ["yes", "sure", "interested", "sounds good", "book", "morning", "afternoon"]):
                                    call_outcome = "interested"

                                # Generate response
                                conversation.append({"role": "user", "content": user_text})
                                is_speaking = True

                                try:
                                    response = await openai_client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=conversation,
                                        max_tokens=120,
                                        temperature=0.7,
                                    )
                                    ai_text = response.choices[0].message.content.strip()
                                except Exception as e:
                                    print(f"[GPT] Error: {e}")
                                    ai_text = "Sorry, I missed that — could you repeat?"

                                print(f"[DODO] {ai_text}")
                                conversation.append({"role": "assistant", "content": ai_text})
                                transcript_log.append(f"Dodo: {ai_text}")

                                await speak(ai_text, stream_sid, ws)
                                is_speaking = False
                            continue

                        # Handle transcript results
                        if dg_msg.get("type") != "Results":
                            continue

                        alt = dg_msg.get("channel", {}).get("alternatives", [{}])[0]
                        transcript = alt.get("transcript", "").strip()
                        is_final   = dg_msg.get("is_final", False)

                        if transcript and is_final:
                            buffer += " " + transcript

                except Exception as e:
                    print(f"[DG→RESPONSE] Error: {e}")

            # Run both loops concurrently
            await asyncio.gather(twilio_to_deepgram(), deepgram_to_response())

    except Exception as e:
        print(f"[WS] Error: {e}")

    # ─── CALL ENDED — send outcome to n8n ───
    print(f"[OUTCOME] {call_outcome}")
    print(f"[TRANSCRIPT] {len(transcript_log)} exchanges")

    if N8N_WEBHOOK_URL:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(N8N_WEBHOOK_URL, json={
                    "outcome":    call_outcome,
                    "transcript": "\n".join(transcript_log),
                    "exchanges":  len(transcript_log),
                }, timeout=10)
            print("[N8N] Outcome sent")
        except Exception as e:
            print(f"[N8N] Failed: {e}")


# ─── TTS → TWILIO ───
async def speak(text: str, stream_sid: str, ws: WebSocket):
    if not stream_sid or not text:
        return

    try:
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="pcm"
        )

        pcm_24k = response.content

        pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
        mulaw     = audioop.lin2ulaw(pcm_8k, 2)

        chunk_size = 160
        for i in range(0, len(mulaw), chunk_size):
            chunk   = mulaw[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("utf-8")
            await ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": stream_sid,
                "media":     {"payload": payload}
            }))
            await asyncio.sleep(0.015)  # Small delay to prevent overwhelming Twilio

    except Exception as e:
        print(f"[TTS] Error: {e}")


# ─── INITIATE OUTBOUND CALL ───
@app.post("/initiate-call")
async def initiate_call(request: Request):
    body          = await request.json()
    to_number     = body.get("to", CALL_TO_NUMBER)
    lead_name     = body.get("name", "there")

    print(f"[TWILIO] Calling {to_number}...")

    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        url=f"{SERVER_URL}/answer",
        method="POST"
    )

    print(f"[TWILIO] Call SID: {call.sid}")
    return {"call_sid": call.sid, "status": call.status}


# ─── START SERVER ───
if __name__ == "__main__":
    print("=" * 50)
    print("EG23 Voice Agent — Full Conversation Mode")
    print(f"SERVER_URL : {SERVER_URL}")
    print("=" * 50)

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
