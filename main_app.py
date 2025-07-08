import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio

from utils.config_manager import config_manager
from utils.logger_config import logger
from llm_service import generate_response_text, load_llm_and_template

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- 전역 변수, 모델, 동시성 제어 ---
CHAT_HISTORY = []
api_lock = asyncio.Lock()  # API 엔드포인트에 대한 동시 접근을 제어하기 위한 Lock
MEMORY_DEPTH = config_manager.getint('llm_params', 'memory_depth', fallback=5)

# --- Pydantic 모델 정의 ---
class ChatRequest(BaseModel):
    user_text: str

def load_models():
    """애플리케이션 시작 시 AI 모델을 로드합니다."""
    logger.info("--- AI VTuber 텍스트 서버 시작 ---")
    logger.info("LLM 모델과 템플릿을 로드합니다. 시간이 걸릴 수 있습니다...")
    try:
        load_llm_and_template()
        logger.info("--- LLM 모델 및 템플릿 로드 완료. 서버가 준비되었습니다. ---")
    except Exception as e:
        logger.error(f"모델 로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise RuntimeError("Failed to load AI models.") from e

@app.on_event("startup")
def startup_event():
    load_models()

def update_chat_history(user_text: str, ai_text: str):
    """대화 기록을 업데이트합니다."""
    CHAT_HISTORY.append({"user": user_text, "ai": ai_text})
    if len(CHAT_HISTORY) > MEMORY_DEPTH:
        CHAT_HISTORY.pop(0)

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """텍스트 입력을 받아 LLM 응답을 반환합니다."""
    async with api_lock:
        try:
            user_text = request.user_text
            if not user_text.strip():
                return JSONResponse(content={"ai_text": ""})

            logger.info(f"[사용자] {user_text}")

            loop = asyncio.get_running_loop()
            
            # LLM 실행
            response_text = await loop.run_in_executor(
                None, generate_response_text, user_text, CHAT_HISTORY
            )
            logger.info(f"[AI 응답] {response_text}")

            update_chat_history(user_text, response_text)
            
            return JSONResponse(content={
                "user_text": user_text,
                "ai_text": response_text,
            })

        except Exception as e:
            logger.error(f"채팅 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

