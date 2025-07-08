from huggingface_hub import hf_hub_download
import os
import logging

logger = logging.getLogger(__name__)

def download_model(repo_id, filename, local_dir):
    """
    Hugging Face Hub에서 모델을 다운로드합니다.
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        logger.info(f"Created directory: {local_dir}")

    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        logger.info(f"Model '{filename}' already exists at '{local_path}'. Skipping download.")
        return local_path

    logger.info(f"Downloading '{filename}' from '{repo_id}' to '{local_dir}'...")
    try:
        # hf_hub_download는 다운로드된 파일의 절대 경로를 반환합니다.
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Windows 호환성
            cache_dir=os.path.join(local_dir, '.cache') # 캐시 위치를 models/.cache로 지정
        )
        
        # 다운로드된 파일이 최종 목적지에 있는지 확인하고, 그렇지 않다면 이동합니다.
        # hf_hub_download는 local_dir에 직접 파일을 저장하려고 시도하지만,
        # 때로는 캐시 경로에 저장될 수 있습니다.
        if not os.path.abspath(downloaded_path) == os.path.abspath(local_path):
             # 심볼릭 링크가 아닌 실제 파일 복사 또는 이동
             # 여기서는 이미 local_dir에 저장되도록 했으므로 이 블록은 안전장치입니다.
             logger.warning(f"Model was downloaded to cache ({downloaded_path}). Moving to {local_path}.")
             # os.rename은 다른 파일 시스템 간에 작동하지 않을 수 있으므로 shutil.move를 사용하는 것이 더 안전합니다.
             import shutil
             shutil.move(downloaded_path, local_path)

        logger.info(f"Model '{filename}' is available at: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"An error occurred during model download: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    from utils.config_manager import config_manager
    from utils.logger_config import logger

    logger.info("--- 모델 다운로더 시작 ---")
    
    # 설정에서 모델 정보 가져오기
    repo_id = config_manager.get('local_llm', 'repo_id')
    model_path = config_manager.get('local_llm', 'model_path')
    
    filename = os.path.basename(model_path)
    local_dir = os.path.dirname(model_path)

    if not all([repo_id, filename, local_dir]):
        logger.error("config.ini의 [local_llm] 섹션에 repo_id와 model_path를 설정해야 합니다.")
    else:
        download_model(repo_id, filename, local_dir)
    
    logger.info("--- 모델 다운로더 종료 ---")