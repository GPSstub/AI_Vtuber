import requests
import sys

SERVER_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{SERVER_URL}/chat/"

def main():
    """서버와 텍스트로 대화하는 간단한 클라이언트입니다."""
    print("--- AI VTuber 텍스트 클라이언트 ---")
    print("채팅을 시작합니다. 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    
    while True:
        try:
            user_text = input("나: ")
            if user_text.lower() in ["exit", "quit"]:
                print("클라이언트를 종료합니다.")
                break

            if not user_text.strip():
                continue

            response = requests.post(CHAT_ENDPOINT, json={"user_text": user_text}, timeout=120)

            if response.status_code == 200:
                data = response.json()
                ai_text = data.get("ai_text", "")
                print(f"AI: {ai_text}")
            else:
                print(f"서버 오류: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"\n[오류] 서버에 연결할 수 없습니다: {e}")
            print("서버가 실행 ���인지 확인하세요.")
            break
        except (KeyboardInterrupt, EOFError):
            print("\n클라이언트를 종료합니다.")
            break

if __name__ == "__main__":
    main()