"""
보약 프로그래밍 언어 파서 (Parser)
토큰 스트림을 AST로 변환합니다.
"""

from typing import List, Optional
from lexer import Token, TokenType, tokenize
from ast_nodes import *


class Parser:
    """보약 파서"""

    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != TokenType.COMMENT]
        self.pos = 0

    def error(self, message: str):
        """파서 오류"""
        token = self.current()
        raise SyntaxError(f"줄 {token.line}: {message} (토큰: {token})")

    def current(self) -> Token:
        """현재 토큰"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def peek(self, offset: int = 0) -> Token:
        """offset만큼 앞의 토큰 확인"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]

    def advance(self) -> Token:
        """다음 토큰으로 이동"""
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def match(self, *types: TokenType) -> bool:
        """현재 토큰이 주어진 타입 중 하나인지 확인"""
        return self.current().type in types

    def expect(self, type: TokenType, message: str = None) -> Token:
        """특정 타입의 토큰 기대"""
        if self.current().type == type:
            return self.advance()
        msg = message or f"'{type.name}' 토큰이 필요합니다"
        self.error(msg)

    def skip_newlines(self):
        """줄바꿈 건너뛰기"""
        while self.match(TokenType.NEWLINE):
            self.advance()

    def parse(self) -> Program:
        """프로그램 파싱"""
        statements = []
        self.skip_newlines()

        while not self.match(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()

        return Program(statements=statements)

    def parse_statement(self) -> Optional[Statement]:
        """문장 파싱"""
        self.skip_newlines()

        if self.match(TokenType.EOF):
            return None

        token = self.current()

        # 조건문: 만약
        if self.match(TokenType.IF):
            return self.parse_if_statement()

        # 컬렉션 반복: 각각의
        if self.match(TokenType.EACH):
            return self.parse_foreach_statement()

        # 무한 반복: 계속 반복하자
        if self.match(TokenType.KEEP):
            return self.parse_infinite_loop()

        # 그만 (break)
        if self.match(TokenType.BREAK):
            self.advance()
            return BreakStatement(line=token.line)

        # 다음으로 (continue)
        if self.match(TokenType.CONTINUE):
            self.advance()
            return ContinueStatement(line=token.line)

        # 시도하자 (try)
        if self.match(TokenType.TRY):
            return self.parse_try_statement()

        # 항상 (상수 선언)
        if self.match(TokenType.ALWAYS):
            return self.parse_constant_declaration()

        # 식별자로 시작하는 문장들
        if self.match(TokenType.IDENTIFIER):
            return self.parse_identifier_statement()

        # 숫자로 시작하는 문장
        if self.match(TokenType.NUMBER):
            return self.parse_number_statement()

        # 문자열로 시작하는 문장
        if self.match(TokenType.STRING):
            return self.parse_string_statement()

        # 그외에 (else) - 보통 if와 함께 처리되지만 독립적으로 나오면 에러
        if self.match(TokenType.ELSE, TokenType.ELIF):
            self.error("조건문 밖에서 '그외에/아니면' 사용")

        return None

    def parse_identifier_statement(self) -> Statement:
        """식별자로 시작하는 문장"""
        start_token = self.current()
        name = self.advance().value  # 식별자

        # 변수명은 값이다 패턴
        if self.match(TokenType.TOPIC_MARKER):  # 은/는
            self.advance()
            value = self.parse_expression()
            if self.match(TokenType.IS):  # 이다
                self.advance()
            return Assignment(
                target=Identifier(name=name, line=start_token.line),
                value=value,
                line=start_token.line
            )

        # 값을 출력하라 패턴
        if self.match(TokenType.OBJECT):  # 을/를
            self.advance()
            if self.match(TokenType.PRINT):  # 출력하라
                self.advance()
                return PrintStatement(
                    value=Identifier(name=name, line=start_token.line),
                    line=start_token.line
                )
            if self.match(TokenType.RETURN):  # 돌려주라
                self.advance()
                return ReturnStatement(
                    value=Identifier(name=name, line=start_token.line),
                    line=start_token.line
                )
            if self.match(TokenType.DEFINE):  # 정의하자
                return self.parse_function_def(name, start_token)

        # 복합 할당: 변수 += 값
        if self.match(TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                      TokenType.MULTIPLY_ASSIGN, TokenType.DIVIDE_ASSIGN):
            op_token = self.advance()
            value = self.parse_expression()
            return CompoundAssignment(
                target=Identifier(name=name, line=start_token.line),
                operator=op_token.value,
                value=value,
                line=start_token.line
            )

        # N부터 M까지 반복하자
        if self.match(TokenType.RANGE_START):  # 부터
            return self.parse_range_loop(
                Identifier(name=name, line=start_token.line),
                start_token
            )

        # 조건인동안 반복하자 (변수가 조건으로 시작)
        # 일반 표현식으로 처리

        # 기타: 표현식 문장
        # 뒤로 돌아가서 전체 표현식 파싱
        self.pos -= 1  # 식별자 위치로 복귀
        expr = self.parse_expression()

        # 표현식 뒤에 을/를 출력하라
        if self.match(TokenType.OBJECT):
            self.advance()
            if self.match(TokenType.PRINT):
                self.advance()
                return PrintStatement(value=expr, line=start_token.line)
            if self.match(TokenType.RETURN):
                self.advance()
                return ReturnStatement(value=expr, line=start_token.line)
            self.pos -= 1  # 조사 위치로 복귀

        return ExpressionStatement(expression=expr, line=start_token.line)

    def parse_number_statement(self) -> Statement:
        """숫자로 시작하는 문장"""
        start_token = self.current()
        num = self.parse_expression()

        # N번 반복하자
        if self.match(TokenType.TIMES):  # 번
            self.advance()
            if self.match(TokenType.LOOP):  # 반복하자
                self.advance()
                return self.parse_times_loop_body(num, start_token)

        # N부터 M까지 반복하자
        if self.match(TokenType.RANGE_START):  # 부터
            return self.parse_range_loop(num, start_token)

        # 숫자를 출력하라
        if self.match(TokenType.OBJECT):
            self.advance()
            if self.match(TokenType.PRINT):
                self.advance()
                return PrintStatement(value=num, line=start_token.line)

        return ExpressionStatement(expression=num, line=start_token.line)

    def parse_string_statement(self) -> Statement:
        """문자열로 시작하는 문장"""
        start_token = self.current()
        string_token = self.advance()

        # 문자열 보간 처리
        if '{' in string_token.value:
            expr = self.parse_interpolated_string(string_token.value, string_token.line)
        else:
            expr = StringLiteral(value=string_token.value, line=string_token.line)

        # "문자열"을 출력하라
        if self.match(TokenType.OBJECT):  # 을/를
            self.advance()
            if self.match(TokenType.PRINT):  # 출력하라
                self.advance()
                return PrintStatement(value=expr, line=start_token.line)
            if self.match(TokenType.INPUT):  # 입력받아라
                self.advance()
                # 이 경우는 프롬프트가 아닌 문자열을 변수에 저장하는 것이 아님
                # 일단 표현식으로 처리
            self.pos -= 1

        return ExpressionStatement(expression=expr, line=start_token.line)

    def parse_expression(self) -> Expression:
        """표현식 파싱"""
        return self.parse_or_expression()

    def parse_or_expression(self) -> Expression:
        """OR 표현식"""
        left = self.parse_and_expression()

        while self.match(TokenType.OR):
            self.advance()
            right = self.parse_and_expression()
            left = BinaryOp(operator='or', left=left, right=right)

        return left

    def parse_and_expression(self) -> Expression:
        """AND 표현식"""
        left = self.parse_not_expression()

        while self.match(TokenType.AND):
            self.advance()
            right = self.parse_not_expression()
            left = BinaryOp(operator='and', left=left, right=right)

        return left

    def parse_not_expression(self) -> Expression:
        """NOT 표현식"""
        if self.match(TokenType.NOT):
            self.advance()
            operand = self.parse_not_expression()
            return UnaryOp(operator='not', operand=operand)
        return self.parse_comparison()

    def parse_comparison(self) -> Expression:
        """비교 표현식"""
        left = self.parse_additive()

        # 한글 비교: "X가 Y보다 크면" 또는 "X가 Y와 같으면"
        if self.match(TokenType.SUBJECT):  # 이/가
            self.advance()
            right = self.parse_additive()

            if self.match(TokenType.THAN):  # 보다
                self.advance()
                if self.match(TokenType.GREATER_EQUAL):
                    self.advance()
                    return BinaryOp(operator='>=', left=left, right=right)
                elif self.match(TokenType.LESS_EQUAL):
                    self.advance()
                    return BinaryOp(operator='<=', left=left, right=right)
                elif self.match(TokenType.GREATER):
                    self.advance()
                    return BinaryOp(operator='>', left=left, right=right)
                elif self.match(TokenType.LESS):
                    self.advance()
                    return BinaryOp(operator='<', left=left, right=right)

            if self.match(TokenType.WITH):  # 과/와
                self.advance()
                if self.match(TokenType.EQUAL):
                    self.advance()
                    return BinaryOp(operator='==', left=left, right=right)
                elif self.match(TokenType.NOT_EQUAL):
                    self.advance()
                    return BinaryOp(operator='!=', left=left, right=right)

        return left

    def parse_additive(self) -> Expression:
        """덧셈/뺄셈"""
        left = self.parse_multiplicative()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = '+' if self.current().type == TokenType.PLUS else '-'
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOp(operator=op, left=left, right=right)

        return left

    def parse_multiplicative(self) -> Expression:
        """곱셈/나눗셈"""
        left = self.parse_power()

        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE,
                         TokenType.FLOOR_DIVIDE, TokenType.MODULO):
            op_map = {
                TokenType.MULTIPLY: '*',
                TokenType.DIVIDE: '/',
                TokenType.FLOOR_DIVIDE: '//',
                TokenType.MODULO: '%',
            }
            op = op_map[self.current().type]
            self.advance()
            right = self.parse_power()
            left = BinaryOp(operator=op, left=left, right=right)

        return left

    def parse_power(self) -> Expression:
        """거듭제곱"""
        left = self.parse_unary()

        if self.match(TokenType.POWER):
            self.advance()
            right = self.parse_power()  # 우결합
            left = BinaryOp(operator='**', left=left, right=right)

        return left

    def parse_unary(self) -> Expression:
        """단항 연산"""
        if self.match(TokenType.MINUS):
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(operator='-', operand=operand)
        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """후위 연산 (인덱스, 속성, 호출)"""
        expr = self.parse_primary()

        while True:
            if self.match(TokenType.LBRACKET):
                # 인덱스 접근
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET, "']'가 필요합니다")
                expr = IndexAccess(target=expr, index=index)

            elif self.match(TokenType.DOT):
                # 속성 접근
                self.advance()
                if self.match(TokenType.IDENTIFIER):
                    attr = self.advance().value
                    expr = AttributeAccess(target=expr, attribute=attr)
                else:
                    self.error("속성 이름이 필요합니다")

            elif self.match(TokenType.LPAREN):
                # 함수 호출
                self.advance()
                args = []
                kwargs = {}

                while not self.match(TokenType.RPAREN):
                    arg = self.parse_expression()
                    args.append(arg)

                    if self.match(TokenType.COMMA):
                        self.advance()

                self.expect(TokenType.RPAREN, "')'가 필요합니다")
                expr = FunctionCall(function=expr, arguments=args, keyword_args=kwargs)

            else:
                break

        return expr

    def parse_primary(self) -> Expression:
        """기본 표현식"""
        token = self.current()

        # 숫자
        if self.match(TokenType.NUMBER):
            self.advance()
            return NumberLiteral(value=token.value, line=token.line)

        # 문자열
        if self.match(TokenType.STRING):
            self.advance()
            # 문자열 보간 처리
            if '{' in token.value:
                return self.parse_interpolated_string(token.value, token.line)
            return StringLiteral(value=token.value, line=token.line)

        # 불리언
        if self.match(TokenType.TRUE):
            self.advance()
            return BooleanLiteral(value=True, line=token.line)

        if self.match(TokenType.FALSE):
            self.advance()
            return BooleanLiteral(value=False, line=token.line)

        # 없음
        if self.match(TokenType.NONE):
            self.advance()
            return NoneLiteral(line=token.line)

        # 식별자
        if self.match(TokenType.IDENTIFIER):
            self.advance()
            return Identifier(name=token.value, line=token.line)

        # 목록
        if self.match(TokenType.LBRACKET):
            return self.parse_list_literal()

        # 사전
        if self.match(TokenType.LBRACE):
            return self.parse_dict_literal()

        # 괄호
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN, "')'가 필요합니다")
            return expr

        self.error(f"예상치 못한 토큰: {token}")

    def parse_interpolated_string(self, template: str, line: int) -> Expression:
        """문자열 보간 파싱"""
        parts = []
        current = ""
        i = 0

        while i < len(template):
            if template[i] == '{':
                if current:
                    parts.append(current)
                    current = ""
                # 변수명 추출
                j = i + 1
                while j < len(template) and template[j] != '}':
                    j += 1
                var_name = template[i+1:j]
                parts.append(Identifier(name=var_name, line=line))
                i = j + 1
            else:
                current += template[i]
                i += 1

        if current:
            parts.append(current)

        return InterpolatedString(parts=parts, line=line)

    def parse_list_literal(self) -> ListLiteral:
        """목록 리터럴"""
        token = self.advance()  # '['
        elements = []

        self.skip_newlines()
        while not self.match(TokenType.RBRACKET):
            elements.append(self.parse_expression())
            self.skip_newlines()
            if self.match(TokenType.COMMA):
                self.advance()
                self.skip_newlines()

        self.expect(TokenType.RBRACKET, "']'가 필요합니다")
        return ListLiteral(elements=elements, line=token.line)

    def parse_dict_literal(self) -> DictLiteral:
        """사전 리터럴"""
        token = self.advance()  # '{'
        pairs = []

        self.skip_newlines()
        while not self.match(TokenType.RBRACE):
            key = self.parse_expression()
            self.expect(TokenType.COLON, "':'가 필요합니다")
            value = self.parse_expression()
            pairs.append((key, value))
            self.skip_newlines()
            if self.match(TokenType.COMMA):
                self.advance()
                self.skip_newlines()

        self.expect(TokenType.RBRACE, "'}'가 필요합니다")
        return DictLiteral(pairs=pairs, line=token.line)

    def parse_if_statement(self) -> IfStatement:
        """조건문 파싱"""
        token = self.advance()  # 만약
        condition = self.parse_expression()

        if self.match(TokenType.THEN):  # 이면/면
            self.advance()

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        then_body = self.parse_block()

        elif_clauses = []
        else_body = None

        while self.match(TokenType.ELIF):  # 아니면
            self.advance()
            elif_condition = self.parse_expression()
            if self.match(TokenType.THEN):
                self.advance()
            self.skip_newlines()
            self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
            elif_body = self.parse_block()
            elif_clauses.append((elif_condition, elif_body))

        if self.match(TokenType.ELSE):  # 그외에
            self.advance()
            self.skip_newlines()
            self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
            else_body = self.parse_block()

        return IfStatement(
            condition=condition,
            then_body=then_body,
            elif_clauses=elif_clauses,
            else_body=else_body,
            line=token.line
        )

    def parse_block(self) -> List[Statement]:
        """블록 파싱"""
        statements = []
        self.skip_newlines()

        while not self.match(TokenType.DEDENT, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()

        if self.match(TokenType.DEDENT):
            self.advance()

        return statements

    def parse_range_loop(self, start_expr: Expression, start_token: Token) -> ForRangeStatement:
        """범위 반복문 파싱"""
        self.advance()  # 부터
        end_expr = self.parse_primary()  # 간단한 숫자/변수만

        inclusive = True
        if self.match(TokenType.RANGE_END):  # 까지
            self.advance()

        step = None
        reverse = False

        if self.match(TokenType.NUMBER):
            step_val = self.advance().value
            if self.match(TokenType.STEP):  # 씩
                self.advance()
                step = NumberLiteral(value=step_val)

        if self.match(TokenType.REVERSE):  # 거꾸로
            reverse = True
            self.advance()

        self.expect(TokenType.LOOP, "'반복하자'가 필요합니다")

        variable = "_i"  # 기본 변수명
        if self.match(TokenType.IDENTIFIER):
            variable = self.advance().value
        if self.match(TokenType.AS):  # (으)로
            self.advance()

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_block()

        return ForRangeStatement(
            variable=variable,
            start=start_expr,
            end=end_expr,
            step=step,
            reverse=reverse,
            inclusive=inclusive,
            body=body,
            line=start_token.line
        )

    def parse_times_loop_body(self, count_expr: Expression, start_token: Token) -> TimesStatement:
        """횟수 반복문 본문 파싱"""
        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_block()

        return TimesStatement(count=count_expr, body=body, line=start_token.line)

    def parse_foreach_statement(self) -> ForEachStatement:
        """컬렉션 반복문"""
        token = self.advance()  # 각각의
        variable = self.expect(TokenType.IDENTIFIER, "변수명이 필요합니다").value

        self.expect(TokenType.OBJECT, "'을/를'이 필요합니다")
        iterable = self.parse_primary()  # 컬렉션 변수
        self.expect(TokenType.FROM, "'에서'가 필요합니다")
        self.expect(TokenType.LOOP, "'반복하자'가 필요합니다")

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_block()

        return ForEachStatement(
            variable=variable,
            iterable=iterable,
            body=body,
            line=token.line
        )

    def parse_infinite_loop(self) -> InfiniteLoop:
        """무한 반복문"""
        token = self.advance()  # 계속
        self.expect(TokenType.LOOP, "'반복하자'가 필요합니다")

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_block()

        return InfiniteLoop(body=body, line=token.line)

    def parse_function_def(self, name: str, start_token: Token) -> FunctionDef:
        """함수 정의 파싱"""
        self.advance()  # 정의하자

        parameters = []
        default_values = {}

        # 매개변수 파싱: "X와 Y를 받아" 또는 "X를 받아"
        if self.match(TokenType.IDENTIFIER):
            while self.match(TokenType.IDENTIFIER):
                param = self.advance().value
                parameters.append(param)

                if self.match(TokenType.WITH):  # 과/와 (여러 매개변수)
                    self.advance()
                elif self.match(TokenType.OBJECT):  # 을/를
                    break

            if self.match(TokenType.OBJECT):
                self.advance()
            if self.match(TokenType.RECEIVE):  # 받아
                self.advance()

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_block()

        return FunctionDef(
            name=name,
            parameters=parameters,
            default_values=default_values,
            body=body,
            line=start_token.line
        )

    def parse_constant_declaration(self) -> Assignment:
        """상수 선언"""
        self.advance()  # 항상
        target_token = self.expect(TokenType.IDENTIFIER, "변수명이 필요합니다")
        self.expect(TokenType.TOPIC_MARKER, "'은/는'이 필요합니다")
        value = self.parse_expression()
        if self.match(TokenType.IS):
            self.advance()

        return Assignment(
            target=Identifier(name=target_token.value, line=target_token.line),
            value=value,
            is_constant=True,
            line=target_token.line
        )

    def parse_try_statement(self) -> TryStatement:
        """예외 처리문"""
        token = self.advance()  # 시도하자

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        try_body = self.parse_block()

        except_clauses = []
        while self.match(TokenType.EXCEPT):  # 오류가나면
            self.advance()
            exc_var = None
            if self.match(TokenType.IDENTIFIER):
                exc_var = self.advance().value
                if self.match(TokenType.AS):
                    self.advance()

            self.skip_newlines()
            self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
            exc_body = self.parse_block()
            except_clauses.append((exc_var, exc_body))

        finally_body = None
        if self.match(TokenType.FINALLY):  # 마무리로
            self.advance()
            self.skip_newlines()
            self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
            finally_body = self.parse_block()

        return TryStatement(
            try_body=try_body,
            except_clauses=except_clauses,
            finally_body=finally_body,
            line=token.line
        )


def parse(source: str) -> Program:
    """소스 코드를 파싱하는 편의 함수"""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()


if __name__ == '__main__':
    test_code = '''
이름은 "홍길동"이다
나이는 25이다

"안녕하세요"를 출력하라
'''
    program = parse(test_code)
    for stmt in program.statements:
        print(stmt)
