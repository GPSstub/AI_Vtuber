# fetch_video_urls.py

import yt_dlp
from utils.logger_config import logger

VIDEO_LIST_FILE = "video_list.txt"
SEARCH_QUERY = "ytsearch100:hololive clips"

def fetch_urls():
    """
    yt-dlp를 사용하여 유튜브에서 영상 URL을 검색하고 파일에 저장합니다.
    """
    logger.info(f"'{SEARCH_QUERY}'로 유튜브 영상 검색을 시작합니다...")
    
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
        'get_url': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(SEARCH_QUERY, download=False)
            
            if 'entries' in result:
                urls = [entry['url'] for entry in result['entries']]
                
                with open(VIDEO_LIST_FILE, 'a', encoding='utf-8') as f:
                    for url in urls:
                        f.write(f"{url}\n")
                
                logger.info(f"{len(urls)}개의 영상 URL을 '{VIDEO_LIST_FILE}'에 추가했습니다.")
            else:
                logger.warning("검색 결과에서 영상을 찾을 수 없습니다.")

    except Exception as e:
        logger.error(f"영상 URL을 가져오는 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    fetch_urls()
