"""
보약 프로그래밍 언어 렉서 (Lexer)
소스 코드를 토큰으로 분리합니다.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class TokenType(Enum):
    """토큰 타입 정의"""
    # 리터럴
    NUMBER = auto()       # 숫자 (정수, 실수)
    STRING = auto()       # 문자열
    TRUE = auto()         # 참
    FALSE = auto()        # 거짓
    NONE = auto()         # 없음

    # 식별자와 키워드
    IDENTIFIER = auto()   # 식별자 (변수명, 함수명)

    # 변수/상수 키워드
    TOPIC_MARKER = auto()    # 은/는
    IS = auto()              # 이다
    ALWAYS = auto()          # 항상

    # 조건문 키워드
    IF = auto()              # 만약
    THEN = auto()            # 이면/면
    ELIF = auto()            # 아니면
    ELSE = auto()            # 그외에

    # 반복문 키워드
    LOOP = auto()            # 반복하자
    WHILE = auto()           # 동안
    EACH = auto()            # 각각의
    FROM = auto()            # 에서
    RANGE_START = auto()     # 부터
    RANGE_END = auto()       # 까지
    TIMES = auto()           # 번
    STEP = auto()            # 씩
    REVERSE = auto()         # 거꾸로
    BREAK = auto()           # 그만
    CONTINUE = auto()        # 다음으로
    KEEP = auto()            # 계속
    AFTER_LOOP = auto()      # 완료후
    AS = auto()              # (으)로 (반복 변수)

    # 함수 키워드
    DEFINE = auto()          # 정의하자
    RECEIVE = auto()         # 받아
    RETURN = auto()          # 돌려주라
    NO_PARAMS = auto()       # 아무것도없이

    # 입출력 키워드
    PRINT = auto()           # 출력하라
    INPUT = auto()           # 입력받아라
    NO_NEWLINE = auto()      # 줄바꿈없이

    # 논리 연산자
    AND = auto()             # 그리고/이고
    OR = auto()              # 또는/이거나
    NOT = auto()             # 아닌

    # 비교 키워드
    EQUAL = auto()           # 같다/같으면
    NOT_EQUAL = auto()       # 다르다/다르면
    THAN = auto()            # 보다
    GREATER = auto()         # 크다/크면
    LESS = auto()            # 작다/작으면
    GREATER_EQUAL = auto()   # 크거나같다/크거나같으면
    LESS_EQUAL = auto()      # 작거나같다/작거나같으면
    IN = auto()              # 에 있으면
    NOT_IN = auto()          # 에 없으면

    # 조사
    SUBJECT = auto()         # 이/가
    OBJECT = auto()          # 을/를
    WITH = auto()            # 과/와
    TO = auto()              # (으)로
    POSSESSIVE = auto()      # 의 (소유격/속성 접근)

    # 예외 처리
    TRY = auto()             # 시도하자
    EXCEPT = auto()          # 오류가나면
    FINALLY = auto()         # 마무리로
    RAISE = auto()           # 발생시켜라

    # 모듈
    IMPORT = auto()          # 가져오라

    # 클래스
    CLASS = auto()           # 틀
    INIT = auto()            # 생성할때
    DESTROY = auto()         # 소멸할때
    SELF = auto()            # 자신
    SELF_ATTR = auto()       # 자신의
    EXTEND = auto()          # 확장하여
    PARENT = auto()          # 부모의
    AT = auto()              # @
    CLASS_METHOD = auto()    # 틀메서드
    STATIC_METHOD = auto()   # 정적메서드

    # 산술 연산자
    PLUS = auto()            # +
    MINUS = auto()           # -
    MULTIPLY = auto()        # *
    DIVIDE = auto()          # /
    FLOOR_DIVIDE = auto()    # //
    MODULO = auto()          # %
    POWER = auto()           # **

    # 할당 연산자
    PLUS_ASSIGN = auto()     # +=
    MINUS_ASSIGN = auto()    # -=
    MULTIPLY_ASSIGN = auto() # *=
    DIVIDE_ASSIGN = auto()   # /=

    # 구분자
    LPAREN = auto()          # (
    RPAREN = auto()          # )
    LBRACKET = auto()        # [
    RBRACKET = auto()        # ]
    LBRACE = auto()          # {
    RBRACE = auto()          # }
    COMMA = auto()           # ,
    COLON = auto()           # :
    DOT = auto()             # .

    # 특수
    NEWLINE = auto()         # 줄바꿈
    INDENT = auto()          # 들여쓰기 증가
    DEDENT = auto()          # 들여쓰기 감소
    EOF = auto()             # 파일 끝
    COMMENT = auto()         # 주석


@dataclass
class Token:
    """토큰 클래스"""
    type: TokenType
    value: any
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


# 한글 키워드 매핑
KEYWORDS = {
    # 값
    '참': TokenType.TRUE,
    '거짓': TokenType.FALSE,
    '없음': TokenType.NONE,

    # 변수/상수
    '은': TokenType.TOPIC_MARKER,
    '는': TokenType.TOPIC_MARKER,
    '이다': TokenType.IS,
    '항상': TokenType.ALWAYS,

    # 조건문
    '만약': TokenType.IF,
    '이면': TokenType.THEN,
    '면': TokenType.THEN,
    '아니면': TokenType.ELIF,
    '그외에': TokenType.ELSE,

    # 반복문
    '반복하자': TokenType.LOOP,
    '동안': TokenType.WHILE,
    '각각의': TokenType.EACH,
    '에서': TokenType.FROM,
    '부터': TokenType.RANGE_START,
    '까지': TokenType.RANGE_END,
    '번': TokenType.TIMES,
    '씩': TokenType.STEP,
    '거꾸로': TokenType.REVERSE,
    '그만': TokenType.BREAK,
    '다음으로': TokenType.CONTINUE,
    '계속': TokenType.KEEP,
    '완료후': TokenType.AFTER_LOOP,
    '로': TokenType.AS,
    '으로': TokenType.AS,

    # 함수
    '정의하자': TokenType.DEFINE,
    '받아': TokenType.RECEIVE,
    '돌려주라': TokenType.RETURN,
    '아무것도없이': TokenType.NO_PARAMS,

    # 입출력
    '출력하라': TokenType.PRINT,
    '입력받아라': TokenType.INPUT,
    '줄바꿈없이': TokenType.NO_NEWLINE,

    # 논리
    '그리고': TokenType.AND,
    '이고': TokenType.AND,
    '또는': TokenType.OR,
    '이거나': TokenType.OR,
    '아닌': TokenType.NOT,

    # 비교 (복합 키워드는 파서에서 처리)
    '같다': TokenType.EQUAL,
    '같으면': TokenType.EQUAL,
    '다르다': TokenType.NOT_EQUAL,
    '다르면': TokenType.NOT_EQUAL,
    '보다': TokenType.THAN,
    '크다': TokenType.GREATER,
    '크면': TokenType.GREATER,
    '작다': TokenType.LESS,
    '작으면': TokenType.LESS,
    '크거나같다': TokenType.GREATER_EQUAL,
    '크거나같으면': TokenType.GREATER_EQUAL,
    '작거나같다': TokenType.LESS_EQUAL,
    '작거나같으면': TokenType.LESS_EQUAL,

    # 조사
    '이': TokenType.SUBJECT,
    '가': TokenType.SUBJECT,
    '을': TokenType.OBJECT,
    '를': TokenType.OBJECT,
    '과': TokenType.WITH,
    '와': TokenType.WITH,

    # 예외
    '시도하자': TokenType.TRY,
    '오류가나면': TokenType.EXCEPT,
    '마무리로': TokenType.FINALLY,
    '발생시켜라': TokenType.RAISE,

    # 모듈
    '가져오라': TokenType.IMPORT,

    # 클래스
    '틀': TokenType.CLASS,
    '생성할때': TokenType.INIT,
    '소멸할때': TokenType.DESTROY,
    '자신': TokenType.SELF,
    '자신의': TokenType.SELF_ATTR,
    '확장하여': TokenType.EXTEND,
    '부모의': TokenType.PARENT,
    '틀메서드': TokenType.CLASS_METHOD,
    '정적메서드': TokenType.STATIC_METHOD,
}


class Lexer:
    """보약 렉서"""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        self.indent_stack = [0]  # 들여쓰기 스택

    def error(self, message: str):
        """렉서 오류"""
        raise SyntaxError(f"줄 {self.line}, 열 {self.column}: {message}")

    def peek(self, offset: int = 0) -> Optional[str]:
        """현재 위치에서 offset만큼 떨어진 문자 확인"""
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None

    def advance(self) -> Optional[str]:
        """다음 문자로 이동"""
        if self.pos < len(self.source):
            char = self.source[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return char
        return None

    def skip_whitespace_same_line(self):
        """같은 줄의 공백 건너뛰기 (줄바꿈 제외)"""
        while self.peek() in (' ', '\t'):
            self.advance()

    def tokenize(self) -> List[Token]:
        """소스 코드를 토큰화"""
        self.tokens = []

        while self.pos < len(self.source):
            # 줄 시작에서 들여쓰기 처리
            if self.column == 1:
                self._handle_indentation()

            char = self.peek()

            # 줄바꿈
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                continue

            # 공백 건너뛰기
            if char in (' ', '\t'):
                self.advance()
                continue

            # 주석
            if char == '#':
                self._handle_comment()
                continue

            # 문자열
            if char in ('"', "'"):
                self._handle_string()
                continue

            # 숫자
            if char.isdigit() or (char == '-' and self.peek(1) and self.peek(1).isdigit()):
                self._handle_number()
                continue

            # 연산자
            if char in '+-*/%':
                self._handle_operator()
                continue

            # 구분자
            if char in '()[]{},:':
                self._handle_delimiter()
                continue

            # 점 (속성 접근 또는 실수)
            if char == '.':
                self.tokens.append(Token(TokenType.DOT, '.', self.line, self.column))
                self.advance()
                continue

            # @ (데코레이터)
            if char == '@':
                self.tokens.append(Token(TokenType.AT, '@', self.line, self.column))
                self.advance()
                continue

            # 한글/영문 식별자 또는 키워드
            if self._is_identifier_start(char):
                self._handle_identifier()
                continue

            self.error(f"알 수 없는 문자: '{char}'")

        # 파일 끝 들여쓰기 정리
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, '', self.line, self.column))

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

    def _handle_indentation(self):
        """들여쓰기 처리"""
        indent = 0
        while self.peek() in (' ', '\t'):
            if self.peek() == ' ':
                indent += 1
            else:  # tab
                indent += 4  # 탭은 4칸으로
            self.advance()

        # 빈 줄이나 주석만 있는 줄은 무시
        if self.peek() in ('\n', '#', None):
            return

        current_indent = self.indent_stack[-1]

        if indent > current_indent:
            self.indent_stack.append(indent)
            self.tokens.append(Token(TokenType.INDENT, indent, self.line, 1))
        elif indent < current_indent:
            while self.indent_stack and self.indent_stack[-1] > indent:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, indent, self.line, 1))
            if self.indent_stack[-1] != indent:
                self.error("잘못된 들여쓰기")

    def _handle_comment(self):
        """주석 처리"""
        self.advance()  # '#' 건너뛰기

        # 여러 줄 주석 확인 (##)
        if self.peek() == '#':
            self.advance()  # 두 번째 '#'
            comment = ''
            while True:
                char = self.advance()
                if char is None:
                    self.error("여러 줄 주석이 닫히지 않았습니다")
                if char == '#' and self.peek() == '#':
                    self.advance()
                    break
                comment += char
            return

        # 한 줄 주석
        while self.peek() and self.peek() != '\n':
            self.advance()

    def _handle_string(self):
        """문자열 처리"""
        quote = self.advance()
        start_line = self.line
        start_col = self.column - 1

        # 삼중 따옴표 확인
        if self.peek() == quote and self.peek(1) == quote:
            self.advance()
            self.advance()
            triple = True
            end_quote = quote * 3
        else:
            triple = False
            end_quote = quote

        string_value = ''
        while True:
            char = self.peek()

            if char is None:
                self.error("문자열이 닫히지 않았습니다")

            if not triple and char == '\n':
                self.error("문자열에서 예기치 않은 줄바꿈")

            if triple:
                if char == quote and self.peek(1) == quote and self.peek(2) == quote:
                    self.advance()
                    self.advance()
                    self.advance()
                    break
            else:
                if char == quote:
                    self.advance()
                    break

            # 이스케이프 시퀀스 처리
            if char == '\\':
                self.advance()
                escaped = self.advance()
                escape_map = {'n': '\n', 't': '\t', '\\': '\\', '"': '"', "'": "'"}
                string_value += escape_map.get(escaped, escaped)
            else:
                string_value += self.advance()

        self.tokens.append(Token(TokenType.STRING, string_value, start_line, start_col))

    def _handle_number(self):
        """숫자 처리"""
        start_col = self.column
        number_str = ''

        # 음수 부호
        if self.peek() == '-':
            number_str += self.advance()

        # 정수 부분
        while self.peek() and (self.peek().isdigit() or self.peek() == '_'):
            char = self.advance()
            if char != '_':  # 천 단위 구분자 무시
                number_str += char

        # 소수점
        if self.peek() == '.' and self.peek(1) and self.peek(1).isdigit():
            number_str += self.advance()  # '.'
            while self.peek() and self.peek().isdigit():
                number_str += self.advance()

        # 숫자로 변환
        if '.' in number_str:
            value = float(number_str)
        else:
            value = int(number_str)

        self.tokens.append(Token(TokenType.NUMBER, value, self.line, start_col))

    def _handle_operator(self):
        """연산자 처리"""
        start_col = self.column
        char = self.advance()
        next_char = self.peek()

        # 복합 연산자
        if next_char == '=':
            self.advance()
            op_map = {
                '+': TokenType.PLUS_ASSIGN,
                '-': TokenType.MINUS_ASSIGN,
                '*': TokenType.MULTIPLY_ASSIGN,
                '/': TokenType.DIVIDE_ASSIGN,
            }
            self.tokens.append(Token(op_map[char], char + '=', self.line, start_col))
            return

        # 거듭제곱 또는 정수 나눗셈
        if char == '*' and next_char == '*':
            self.advance()
            self.tokens.append(Token(TokenType.POWER, '**', self.line, start_col))
            return
        if char == '/' and next_char == '/':
            self.advance()
            self.tokens.append(Token(TokenType.FLOOR_DIVIDE, '//', self.line, start_col))
            return

        # 단일 연산자
        op_map = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
        }
        self.tokens.append(Token(op_map[char], char, self.line, start_col))

    def _handle_delimiter(self):
        """구분자 처리"""
        char = self.advance()
        delim_map = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ',': TokenType.COMMA,
            ':': TokenType.COLON,
        }
        self.tokens.append(Token(delim_map[char], char, self.line, self.column - 1))

    def _is_identifier_start(self, char: str) -> bool:
        """식별자 시작 문자인지 확인"""
        if char is None:
            return False
        # 한글, 영문, 밑줄
        return (
            '\uAC00' <= char <= '\uD7A3' or  # 한글 음절
            '\u1100' <= char <= '\u11FF' or  # 한글 자모
            '\u3130' <= char <= '\u318F' or  # 한글 호환 자모
            char.isalpha() or
            char == '_'
        )

    def _is_identifier_char(self, char: str) -> bool:
        """식별자 문자인지 확인"""
        return self._is_identifier_start(char) or (char and char.isdigit())

    def _handle_identifier(self):
        """식별자/키워드 처리"""
        start_col = self.column
        identifier = ''

        while self._is_identifier_char(self.peek()):
            identifier += self.advance()

        # 키워드 확인
        if identifier in KEYWORDS:
            self.tokens.append(Token(KEYWORDS[identifier], identifier, self.line, start_col))
        else:
            # 식별자 끝에 붙은 조사 분리
            # 예: "이름은" -> "이름" + "은"
            # 예: "나이가" -> "나이" + "가"
            suffix_map = {
                '은': TokenType.TOPIC_MARKER,
                '는': TokenType.TOPIC_MARKER,
                '이': TokenType.SUBJECT,
                '가': TokenType.SUBJECT,
                '을': TokenType.OBJECT,
                '를': TokenType.OBJECT,
                '과': TokenType.WITH,
                '와': TokenType.WITH,
                '로': TokenType.AS,
                '의': TokenType.POSSESSIVE,
            }

            # 긴 접미사부터 확인 (복합 조사)
            found_suffix = False
            for suffix_len in [3, 2, 1]:
                if len(identifier) > suffix_len:
                    suffix = identifier[-suffix_len:]
                    base = identifier[:-suffix_len]

                    # base가 키워드이면 항상 분리 (예: "틀을" -> "틀" + "을")
                    # base가 키워드가 아니고 너무 짧으면 분리하지 않음
                    # "이/가"는 단어 끝에 자주 오므로 base가 3자 이상일 때만 분리
                    # 예: "나이"를 "나" + "이"로 분리하지 않음
                    # 예: "고양이"를 "고양" + "이"로 분리하지 않음
                    # 다른 접미사는 base가 2자 이상일 때 분리
                    if base not in KEYWORDS:
                        # 이/가는 한국어 단어 끝에 자주 나오므로 더 엄격하게
                        if suffix in ('이', '가') and len(base) < 3:
                            continue
                        # 다른 접미사는 base가 2자 이상이면 분리
                        if len(base) < 2:
                            continue

                    if suffix in KEYWORDS:
                        # base가 키워드인지 확인
                        if base in KEYWORDS:
                            self.tokens.append(Token(KEYWORDS[base], base, self.line, start_col))
                        else:
                            self.tokens.append(Token(TokenType.IDENTIFIER, base, self.line, start_col))
                        self.tokens.append(Token(KEYWORDS[suffix], suffix, self.line, start_col + len(base)))
                        found_suffix = True
                        break
                    elif suffix in suffix_map:
                        # base가 키워드인지 확인
                        if base in KEYWORDS:
                            self.tokens.append(Token(KEYWORDS[base], base, self.line, start_col))
                        else:
                            self.tokens.append(Token(TokenType.IDENTIFIER, base, self.line, start_col))
                        self.tokens.append(Token(suffix_map[suffix], suffix, self.line, start_col + len(base)))
                        found_suffix = True
                        break

            if not found_suffix:
                self.tokens.append(Token(TokenType.IDENTIFIER, identifier, self.line, start_col))


def tokenize(source: str) -> List[Token]:
    """소스 코드를 토큰화하는 편의 함수"""
    lexer = Lexer(source)
    return lexer.tokenize()


if __name__ == '__main__':
    # 테스트
    test_code = '''
# 변수 선언 테스트
이름은 "홍길동"이다
나이는 25이다

만약 나이가 18보다 크면
    "성인입니다"를 출력하라
그외에
    "미성년자입니다"를 출력하라
'''
    tokens = tokenize(test_code)
    for token in tokens:
        print(token)
