#!/usr/bin/env python3
"""
보약 프로그래밍 언어 - 메인 실행 파일

사용법:
    python boyak.py              # 대화형 모드 (REPL)
    python boyak.py 파일.보약    # 파일 실행
"""

import sys
import os

# 현재 디렉토리를 모듈 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interpreter import Interpreter, BoyakError, repl, run_file


def main():
    """메인 함수"""
    if len(sys.argv) == 1:
        # 대화형 모드
        repl()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]

        if filename in ['-h', '--help', '도움말']:
            print_help()
        elif filename in ['-v', '--version', '버전']:
            print_version()
        else:
            # 파일 실행
            if not os.path.exists(filename):
                print(f"오류: 파일을 찾을 수 없습니다: {filename}")
                sys.exit(1)
            run_file(filename)
    else:
        print("사용법: boyak [파일명.보약]")
        sys.exit(1)


def print_help():
    """도움말 출력"""
    print("""
보약 프로그래밍 언어 v1.0

사용법:
    boyak                    대화형 모드 (REPL) 시작
    boyak <파일.보약>        보약 소스 파일 실행
    boyak --help            이 도움말 표시
    boyak --version         버전 정보 표시

예시:
    boyak 안녕.보약          '안녕.보약' 파일 실행
    boyak examples/계산기.보약

자세한 정보는 docs/ 폴더의 문서를 참조하세요.
""")


def print_version():
    """버전 정보 출력"""
    print("""
보약 (Boyak) 프로그래밍 언어
버전: 1.0.0
한글 기반의 프로그래밍 언어

GitHub: https://github.com/boyak-lang/boyak
라이선스: MIT
""")


if __name__ == '__main__':
    main()
