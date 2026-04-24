"""
EG23 Voice Agent — Simple Test
Just run this script, it calls your phone, and Dodo speaks.
Requirements: pip install twilio openai fastapi uvicorn python-dotenv
"""

import os
import asyncio
import base64
import json
import audioop
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.rest import Client
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG — fill these in or add to .env ───
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "ACxxxxxxxx")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",   "xxxxxxxx")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER", "+1xxxxxxxxxx")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY",      "sk-xxxxxxxx")
SERVER_URL          = os.getenv("SERVER_URL",           "https://xxxx.ngrok.io")
CALL_TO_NUMBER      = os.getenv("CALL_TO_NUMBER",       "+1xxxxxxxxxx")  # YOUR phone number

openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()


# ─────────────────────────────────────────────
# STEP 1 — Twilio calls this to get instructions
# ─────────────────────────────────────────────
@app.post("/answer")
async def answer(request: Request):
    """
    When the call connects, Twilio hits this endpoint.
    We tell Twilio to open a media stream to our WebSocket.
    """
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">Please wait while we connect you to Dodo.</Say>
  <Connect>
    <Stream url="wss://{SERVER_URL.replace('https://','').replace('http://','')}/stream"/>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# ─────────────────────────────────────────────
# STEP 2 — WebSocket handles live audio
# ─────────────────────────────────────────────
@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[WS] Twilio connected")

    stream_sid = None
    has_spoken_opening = False

    try:
        async for raw in ws.iter_text():
            msg   = json.loads(raw)
            event = msg.get("event")

            # ── Stream started ──
            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                print(f"[STREAM] Started: {stream_sid}")

                # Speak opening message immediately
                if not has_spoken_opening:
                    has_spoken_opening = True
                    opening = (
                        "Hey — this is Dodo calling from EG23. "
                        "I'm an AI assistant. Just calling to confirm you'd like to try "
                        "our free AI automation setup. Can you hear me okay?"
                    )
                    print(f"[DODO] {opening}")
                    await speak(opening, stream_sid, ws)

            # ── Incoming audio from caller ──
            elif event == "media":
                # For this simple test we just echo a response after receiving audio
                # In the full version this goes to Deepgram for STT
                pass

            elif event == "stop":
                print("[STREAM] Stopped")
                break

    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


# ─────────────────────────────────────────────
# STEP 3 — TTS using OpenAI → stream to Twilio
# ─────────────────────────────────────────────
async def speak(text: str, stream_sid: str, ws: WebSocket):
    """Convert text to speech and stream audio back to the caller via Twilio."""
    print(f"[TTS] Generating: {text[:60]}...")

    try:
        # Generate speech with OpenAI
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",          # warm, friendly voice
            input=text,
            response_format="pcm"  # raw PCM 24kHz
        )

        pcm_24k = response.content

        # Convert PCM 24kHz stereo → mulaw 8kHz mono (Twilio format)
        pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
        mulaw      = audioop.lin2ulaw(pcm_8k, 2)

        # Send audio in 20ms chunks
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


# ─────────────────────────────────────────────
# STEP 4 — Initiate the outbound call
# ─────────────────────────────────────────────
def make_call():
    print(f"[TWILIO] Calling {CALL_TO_NUMBER}...")

    call = twilio_client.calls.create(
        to=CALL_TO_NUMBER,
        from_=TWILIO_FROM_NUMBER,
        url=f"{SERVER_URL}/answer",   # Twilio fetches TwiML from here
        method="POST"
    )

    print(f"[TWILIO] Call initiated → SID: {call.sid}")
    print(f"[TWILIO] Status: {call.status}")
    return call.sid


# ─────────────────────────────────────────────
# MAIN — Start server then make the call
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import threading
    import time

    print("=" * 50)
    print("EG23 Voice Agent — Simple Test")
    print("=" * 50)
    print(f"Server URL : {SERVER_URL}")
    print(f"Calling    : {CALL_TO_NUMBER}")
    print("=" * 50)

    # Start the FastAPI server in a background thread
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(2)
    print("[SERVER] Running on port 8000")

    # Make the call
    make_call()

    print("\n[WAITING] Keep this running until the call ends...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOPPED]")
