from llama_cpp import Llama
from utils.logger_config import logger
from utils.config_manager import config_manager
from typing import List, Dict
import jinja2
import os

# --- LLM 및 템플릿 설정 로드 ---
LOCAL_LLM_MODEL_PATH = config_manager.get('local_llm', 'model_path')
N_GPU_LAYERS = config_manager.getint('local_llm', 'n_gpu_layers', fallback=0)

LLM_MAX_TOKENS = config_manager.getint('llm_params', 'max_tokens', fallback=256)
LLM_TEMPERATURE = config_manager.getfloat('llm_params', 'temperature', fallback=0.7)
LLM_TOP_P = config_manager.getfloat('llm_params', 'top_p', fallback=0.95)
LLM_TOP_K = config_manager.getint('llm_params', 'top_k', fallback=40)
LLM_REPEAT_PENALTY = config_manager.getfloat('llm_params', 'repeat_penalty', fallback=1.1)

PROMPT_TEMPLATE_PATH = config_manager.get('templates', 'prompt_template_path')

llm_model = None
prompt_template = None

def load_llm_and_template():
    """LLM 모델과 프롬프트 템플릿을 로드합니다."""
    global llm_model, prompt_template
    
    # LLM 모델 로드
    if llm_model is None:
        logger.info("로컬 LLM 모델 로드 중... (시간이 오래 걸릴 수 있습니다)")
        try:
            if not os.path.exists(LOCAL_LLM_MODEL_PATH):
                raise FileNotFoundError(f"LLM model not found at: {LOCAL_LLM_MODEL_PATH}")

            logger.info(f"Loading model from: {LOCAL_LLM_MODEL_PATH}")
            llm_model = Llama(
                model_path=LOCAL_LLM_MODEL_PATH,
                n_gpu_layers=N_GPU_LAYERS,
                n_ctx=config_manager.getint('local_llm', 'n_ctx', fallback=2048),
                verbose=False
            )
            logger.info("로컬 LLM 모델 로드 완료.")
        except Exception as e:
            logger.error(f"로컬 LLM 모델 로드 중 오류 발생: {e}")
            logger.error("GPU 설정(n_gpu_layers)을 확인하거나, 모델 파일 경로를 확인해주세요.")
            exit()

    # 프롬프트 템플릿 로드
    if prompt_template is None:
        try:
            template_dir = os.path.dirname(os.path.abspath(PROMPT_TEMPLATE_PATH))
            template_name = os.path.basename(PROMPT_TEMPLATE_PATH)
            env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
            prompt_template = env.get_template(template_name)
            logger.info(f"프롬프트 템플릿 로드 완료: {PROMPT_TEMPLATE_PATH}")
        except Exception as e:
            logger.error(f"프롬프트 템플릿 로드 중 오류 발생: {e}")
            exit()

def generate_response_text(user_text: str, chat_history: List[Dict[str, str]]) -> str:
    """로컬 LLM을 사용하여 사용자의 말과 이전 대화 기록에 대한 응답을 생성합니다."""
    load_llm_and_template()
    logger.info("로컬 LLM으로 응답 생성 중...")

    # --- 템플릿을 사용하여 프롬프트 렌더링 ---
    prompt = prompt_template.render(
        chat_history=chat_history,
        user_text=user_text
    )
    
    try:
        output = llm_model(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            stop=["###", "\nUser:", "\nYou:"],
            echo=False,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            repeat_penalty=LLM_REPEAT_PENALTY
        )
        
        response_text = output["choices"][0]["text"].strip()
        logger.info(f"AI 응답 (텍스트): {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"로컬 LLM 응답 생성 중 오류 발생: {e}")
        return "죄송해요, 지금은 답변을 생각하기가 어렵네요."

