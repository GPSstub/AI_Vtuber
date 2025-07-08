import os
import json
import yt_dlp
from pyannote.audio import Pipeline
import torch
import torchaudio
import whisper
from utils.logger_config import logger
from imageio_ffmpeg import get_ffmpeg_exe
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 설정 ---
VIDEO_LIST_FILE = "video_list.txt"
OUTPUT_DIR = "training_data"
TRANSCRIPT_DIR = os.path.join(OUTPUT_DIR, "transcripts")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
FINAL_TRAINING_FILE = os.path.join(OUTPUT_DIR, "training_data.jsonl")
FFMPEG_PATH = get_ffmpeg_exe()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- 디렉토리 생성 ---
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

def download_audio(url, output_path):
    """유튜브 영상에서 음원을 추출하여 저장합니다."""
    logger.info(f"'{url}'에서 음원 다운로드 중...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'ffmpeg_location': FFMPEG_PATH,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"음원 다운로드 완료: {output_path}")
        return True
    except Exception as e:
        logger.error(f"음원 다운로드 중 오류 발생: {e}")
        return False

import soundfile as sf

def transcribe_audio(audio_path):
    """Whisper를 사용하여 음성 파일을 텍스트로 변환하고 타임스탬프를 기록합니다."""
    logger.info(f"'{audio_path}'의 음성을 텍스트로 변환 중...")
    model_dir = os.path.join(OUTPUT_DIR, "whisper_models")
    os.makedirs(model_dir, exist_ok=True)
    try:
        audio, sample_rate = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(torch.tensor(audio, dtype=torch.float32)).numpy()

        model = whisper.load_model("base", download_root=model_dir)
        result = model.transcribe(audio, verbose=False, language='ko')
        logger.info("음성 텍스트 변환 완료.")
        return result['segments']
    except Exception as e:
        logger.error(f"음성 텍스트 변환 중 오류 발생: {e}", exc_info=True)
        return None

def diarize_speakers(audio_path):
    """pyannote.audio를 사용하여 화자를 분리합니다."""
    if not HF_TOKEN:
        logger.error("Hugging Face 토큰(HF_TOKEN)이 설정되지 않았습니다. .env 파일이나 환경 변수를 확인하세요.")
        return None
        
    logger.info(f"'{audio_path}'에서 화자 분리 중...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        # MPS 장치가 사용 가능한 경우 사용
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            pipeline.to(device)
            
        diarization = pipeline(audio_path)
        logger.info("화자 분리 완료.")
        return diarization
    except Exception as e:
        logger.error(f"화자 분리 중 오류 발생: {e}", exc_info=True)
        return None

def sanitize_filename(url):
    """URL에서 영상 ID를 추출하고 안전한 파일 이름으로 만듭니다."""
    video_id = url.split("=")[-1]
    if "/" in video_id:
        video_id = video_id.split("/")[-1]
    return "".join(c for c in video_id if c.isalnum() or c in ('-', '_'))

def create_training_data():
    """전체 데이터 생성 파이프라인을 실행합니다."""
    logger.info("학습 데이터 생성을 시작합니다.")

    if not os.path.exists(VIDEO_LIST_FILE):
        logger.error(f"영상 목록 파일 '{VIDEO_LIST_FILE}'을 찾을 수 없습니다.")
        with open(VIDEO_LIST_FILE, 'w', encoding='utf-8') as f:
            f.write("# 여기에 유튜브 영상 URL을 한 줄에 하나씩 입력하세요.\n")
        return

    with open(VIDEO_LIST_FILE, 'r', encoding='utf-8') as f:
        video_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    with open(FINAL_TRAINING_FILE, 'w', encoding='utf-8') as final_file:
        for url in video_urls:
            video_id = sanitize_filename(url)
            audio_filename = f"{video_id}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            
            # 파일이 이미 존재하면 건너뛰기
            if os.path.exists(audio_path):
                logger.info(f"이미 처리된 파일입니다: {audio_path}")
            else:
                if not download_audio(url, audio_path.replace('.wav', '')):
                    continue

            transcription = transcribe_audio(audio_path)
            if not transcription:
                continue

            diarization = diarize_speakers(audio_path)
            if not diarization:
                continue
            
            # 화자와 대사 매칭
            dialogues = []
            for segment in transcription:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']

                # 해당 세그먼트 시간과 겹치는 화자 찾기
                segment_speakers = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start < end_time and turn.end > start_time:
                        segment_speakers.append(speaker)
                
                # 가장 가능성 있는 화자 선택 (여기서는 간단히 첫 번째 화자 선택)
                # 또는 화자가 없거나 여러 명인 경우 'Unknown'으로 처리
                if len(set(segment_speakers)) == 1:
                    speaker_id = segment_speakers[0]
                else:
                    speaker_id = "Unknown" 
                
                dialogues.append({"speaker": speaker_id, "text": text})

            # 최종 데이터 구조화 및 저장
            training_entry = {
                "video_id": video_id,
                "url": url,
                "dialogues": dialogues
            }
            
            final_file.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
            logger.info(f"'{url}' 영상 처리를 완료하고 결과를 저장했습니다.")

    logger.info(f"모든 영상 처리가 완료되었습니다. 최종 파일: {FINAL_TRAINING_FILE}")

if __name__ == "__main__":
    create_training_data()

