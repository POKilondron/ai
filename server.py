from fastapi import FastAPI, Request
from gradio_client import Client
import uvicorn

app = FastAPI()

client = Client("POKuuuu/675")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    result = client.predict(
        message=user_message,
        system_message="You are a friendly Chatbot.",
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        api_name="/chat"
    )

    return {"response": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

