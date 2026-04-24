"""
EG23 Voice Agent — Simple Test
Just run this script, it calls your phone, and Dodo speaks.
"""

import os
import asyncio
import base64
import json
import struct
import array
import threading
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.rest import Client
from openai import OpenAI
import uvicorn
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
            """Resample audio data."""
            n_samples_in  = len(data) // (width * nchannels)
            n_samples_out = int(n_samples_in * outrate / inrate)
            result = bytearray()
            for i in range(n_samples_out):
                src = int(i * inrate / outrate)
                src = min(src, n_samples_in - 1)
                for c in range(nchannels):
                    offset = (src * nchannels + c) * width
                    chunk  = data[offset:offset + width]
                    result.extend(chunk)
            return bytes(result), None

        @staticmethod
        def lin2ulaw(data, width):
            """Convert linear PCM to u-law."""
            fmt     = '<h' if width == 2 else '<b'
            n       = len(data) // width
            result  = []
            for i in range(n):
                s = struct.unpack_from(fmt, data, i * width)[0]
                if width == 1:
                    s = s * 256
                s    = max(-32768, min(32767, s))
                sign = 0x80 if s < 0 else 0
                s    = abs(s)
                s   += 132
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
SERVER_URL          = os.getenv("SERVER_URL", "").rstrip("/")
CALL_TO_NUMBER      = os.getenv("CALL_TO_NUMBER")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()


# ─── HEALTH CHECK ───
@app.get("/")
async def health():
    return {"status": "EG23 Voice Agent running"}


# ─── TWILIO ANSWER WEBHOOK ───
@app.post("/answer")
async def answer(request: Request):
    ws_url = SERVER_URL.replace("https://", "wss://").replace("http://", "ws://")
    twiml  = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">Please wait while we connect you.</Say>
  <Connect>
    <Stream url="{ws_url}/stream"/>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# ─── WEBSOCKET AUDIO HANDLER ───
@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[WS] Twilio connected")

    stream_sid         = None
    has_spoken_opening = False

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
                        "Hey — this is Dodo calling from EG23. "
                        "I'm an AI assistant. I'm calling to confirm you'd like to try "
                        "our free AI automation setup. Can you hear me okay?"
                    )
                    print(f"[DODO] {opening}")
                    await speak(opening, stream_sid, ws)

            elif event == "media":
                pass  # STT goes here in full version

            elif event == "stop":
                print("[STREAM] Stopped")
                break

    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


# ─── TTS → TWILIO ───
async def speak(text: str, stream_sid: str, ws: WebSocket):
    print(f"[TTS] Generating speech...")
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="pcm"
        )

        pcm_24k = response.content

        # Resample 24000 → 8000
        pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)

        # Convert to mulaw
        mulaw = audioop.lin2ulaw(pcm_8k, 2)

        # Stream to Twilio in 20ms chunks
        chunk_size = 160
        for i in range(0, len(mulaw), chunk_size):
            chunk   = mulaw[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("utf-8")
            await ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": stream_sid,
                "media":     {"payload": payload}
            }))

        print("[TTS] Done streaming audio")

    except Exception as e:
        print(f"[TTS] Error: {e}")


# ─── INITIATE OUTBOUND CALL ───
@app.post("/initiate-call")
async def initiate_call(request: Request):
    body          = await request.json()
    to_number     = body.get("to", CALL_TO_NUMBER)
    lead_name     = body.get("name", "there")
    business_type = body.get("business_type", "")
    goal          = body.get("goal", "")

    print(f"[TWILIO] Calling {to_number}...")

    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        url=f"{SERVER_URL}/answer",
        method="POST"
    )

    print(f"[TWILIO] Call SID: {call.sid} | Status: {call.status}")
    return {"call_sid": call.sid, "status": call.status}


# ─── START SERVER ───
if __name__ == "__main__":
    print("=" * 50)
    print("EG23 Voice Agent — Test Mode")
    print(f"SERVER_URL : {SERVER_URL}")
    print(f"Calling    : {CALL_TO_NUMBER}")
    print("=" * 50)

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
