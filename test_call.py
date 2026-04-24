"""
EG23 Voice Agent — Groq Edition
Stack: Twilio + Deepgram (STT) + Groq Llama 3.1 + OpenAI TTS
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
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
SERVER_URL          = os.getenv("SERVER_URL", "").rstrip("/")
CALL_TO_NUMBER      = os.getenv("CALL_TO_NUMBER")
N8N_WEBHOOK_URL     = os.getenv("N8N_WEBHOOK_URL", "")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
groq_client   = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()


# ─── DODO SYSTEM PROMPT ───
DODO_PROMPT = """You are Dodo — EG23's AI sales assistant calling a lead who requested a free trial on eg23.net.

EG23 builds AI agents, workflows, and automations for small businesses.

YOUR GOAL
1. Confirm the right person
2. Find out what they want to automate
3. Book a 15-minute call with the EG23 team

STYLE — CRITICAL FOR VOICE
- Speak like a real person on the phone
- Very short sentences. One idea at a time.
- Use natural phrases: "okay", "got it", "right", "makes sense"
- Never use lists, bullet points, or formal language
- Responses MUST be under 20 words unless answering a specific question
- Never say "I'm an AI" — you are Dodo

RULES
- If they're busy: "Totally get it — want me to call back later?"
- If not interested: "No worries, thanks for picking up. Have a good one."
- Pricing: "Depends on your setup — most start under a hundred for setup. The team covers it on the call."
- Never invent features or guarantees

Keep it flowing naturally. Be brief. Sound human."""


@app.get("/")
async def health():
    return {"status": "EG23 Voice Agent running", "version": "groq-v2"}


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


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[WS] Twilio connected")

    stream_sid         = None
    conversation       = [{"role": "system", "content": DODO_PROMPT}]
    transcript_log     = []
    call_outcome       = "unknown"
    is_speaking        = False
    has_spoken_opening = False

    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?encoding=mulaw"
        "&sample_rate=8000"
        "&channels=1"
        "&model=nova-2"
        "&interim_results=true"
        "&utterance_end_ms=1000"
        "&vad_events=true"
        "&endpointing=300"
    )
    dg_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    try:
        async with websockets.connect(dg_url, additional_headers=dg_headers) as dg_ws:
            print("[DEEPGRAM STT] Connected")

            async def twilio_to_deepgram():
                nonlocal stream_sid, has_spoken_opening
                try:
                    async for raw in ws.iter_text():
                        msg   = json.loads(raw)
                        event = msg.get("event")

                        if event == "start":
                            stream_sid = msg["start"]["streamSid"]
                            print(f"[STREAM] Started: {stream_sid}")

                            if not has_spoken_opening:
                                has_spoken_opening = True
                                opening = (
                                    "Hey, this is Dodo from EG23. "
                                    "You requested a free trial on our site. Got two minutes?"
                                )
                                print(f"[DODO] {opening}")
                                conversation.append({"role": "assistant", "content": opening})
                                transcript_log.append(f"Dodo: {opening}")
                                await speak_openai(opening, stream_sid, ws)

                        elif event == "media":
                            audio_b64   = msg["media"]["payload"]
                            audio_bytes = base64.b64decode(audio_b64)
                            await dg_ws.send(audio_bytes)

                        elif event == "stop":
                            print("[STREAM] Stopped")
                            break

                except WebSocketDisconnect:
                    print("[WS] Twilio disconnected")
                except Exception as e:
                    print(f"[TWILIO->DG] Error: {e}")

            async def deepgram_to_response():
                nonlocal is_speaking, call_outcome
                buffer = ""

                try:
                    async for raw in dg_ws:
                        dg_msg = json.loads(raw)

                        if dg_msg.get("type") == "UtteranceEnd":
                            if buffer.strip() and not is_speaking:
                                user_text = buffer.strip()
                                buffer = ""

                                print(f"[USER] {user_text}")
                                transcript_log.append(f"Lead: {user_text}")

                                lower = user_text.lower()
                                if any(w in lower for w in ["not interested", "no thanks", "stop calling"]):
                                    call_outcome = "not_interested"
                                elif any(w in lower for w in ["yes", "sure", "interested", "sounds good", "book"]):
                                    call_outcome = "interested"

                                conversation.append({"role": "user", "content": user_text})
                                is_speaking = True

                                await stream_groq_and_speak(conversation, stream_sid, ws, transcript_log)
                                is_speaking = False
                            continue

                        if dg_msg.get("type") != "Results":
                            continue

                        alt        = dg_msg.get("channel", {}).get("alternatives", [{}])[0]
                        transcript = alt.get("transcript", "").strip()
                        is_final   = dg_msg.get("is_final", False)

                        if transcript and is_final:
                            buffer += " " + transcript

                except Exception as e:
                    print(f"[DG->RESPONSE] Error: {e}")

            await asyncio.gather(twilio_to_deepgram(), deepgram_to_response())

    except Exception as e:
        print(f"[WS] Error: {e}")

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


async def stream_groq_and_speak(conversation, stream_sid, ws, transcript_log):
    full_response = ""
    sentence_buf  = ""

    try:
        stream = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation,
            max_tokens=100,
            temperature=0.7,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            sentence_buf  += delta
            full_response += delta

            if any(p in sentence_buf for p in ['.', '!', '?']):
                last_punct = max(
                    sentence_buf.rfind('.'),
                    sentence_buf.rfind('!'),
                    sentence_buf.rfind('?')
                )
                if last_punct > 0:
                    to_speak     = sentence_buf[:last_punct + 1].strip()
                    sentence_buf = sentence_buf[last_punct + 1:]
                    if to_speak:
                        await speak_openai(to_speak, stream_sid, ws)

        if sentence_buf.strip():
            await speak_openai(sentence_buf.strip(), stream_sid, ws)

        full_response = full_response.strip()
        print(f"[DODO] {full_response}")
        conversation.append({"role": "assistant", "content": full_response})
        transcript_log.append(f"Dodo: {full_response}")

    except Exception as e:
        print(f"[GROQ] Error: {e}")
        fallback = "Sorry, I missed that. Could you repeat?"
        await speak_openai(fallback, stream_sid, ws)


async def speak_openai(text: str, stream_sid: str, ws: WebSocket):
    if not stream_sid or not text:
        return

    try:
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="pcm"
        )

        pcm_24k   = response.content
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
            await asyncio.sleep(0.018)

    except Exception as e:
        print(f"[TTS] Error: {e}")


@app.post("/initiate-call")
async def initiate_call(request: Request):
    body      = await request.json()
    to_number = body.get("to", CALL_TO_NUMBER)

    print(f"[TWILIO] Calling {to_number}...")

    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        url=f"{SERVER_URL}/answer",
        method="POST"
    )

    print(f"[TWILIO] Call SID: {call.sid}")
    return {"call_sid": call.sid, "status": call.status}


if __name__ == "__main__":
    print("=" * 50)
    print("EG23 Voice Agent — Groq v2")
    print(f"SERVER_URL : {SERVER_URL}")
    print("=" * 50)

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
