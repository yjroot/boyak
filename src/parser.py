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

        # 자신의로 시작하는 문장 (메서드 내부)
        if self.match(TokenType.SELF_ATTR):
            return self.parse_self_statement()

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

        # 클래스 정의: "사람 틀을 정의하자"
        if self.match(TokenType.CLASS):  # 틀
            self.advance()
            if self.match(TokenType.OBJECT):  # 을
                self.advance()
                if self.match(TokenType.DEFINE):  # 정의하자
                    return self.parse_class_def(name, start_token)

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

    def parse_self_statement(self) -> Statement:
        """자신의로 시작하는 문장 (메서드 내부)"""
        start_token = self.current()
        self.advance()  # 자신의 소비

        # 속성 이름
        attr_token = self.expect(TokenType.IDENTIFIER, "속성 이름이 필요합니다")
        attr_name = attr_token.value
        self_access = SelfAccess(attribute=attr_name, line=start_token.line)

        # 자신의 X는 Y이다 (속성 할당)
        if self.match(TokenType.TOPIC_MARKER):
            self.advance()
            value = self.parse_expression()
            if self.match(TokenType.IS):
                self.advance()
            return Assignment(
                target=self_access,
                value=value,
                line=start_token.line
            )

        # 자신의 X를 출력하라 / 돌려주라
        if self.match(TokenType.OBJECT):
            self.advance()
            if self.match(TokenType.PRINT):
                self.advance()
                return PrintStatement(value=self_access, line=start_token.line)
            if self.match(TokenType.RETURN):
                self.advance()
                return ReturnStatement(value=self_access, line=start_token.line)
            self.pos -= 1  # 조사 위치로 복귀

        # 표현식 문장 (자신의 X + Y 등)
        # 다른 연산이 있을 수 있으므로 표현식 파싱 계속
        self.pos = start_token.pos if hasattr(start_token, 'pos') else self.pos - 2
        expr = self.parse_expression()

        # 표현식 뒤에 을/를 출력하라/돌려주라
        if self.match(TokenType.OBJECT):
            self.advance()
            if self.match(TokenType.PRINT):
                self.advance()
                return PrintStatement(value=expr, line=start_token.line)
            if self.match(TokenType.RETURN):
                self.advance()
                return ReturnStatement(value=expr, line=start_token.line)
            self.pos -= 1

        return ExpressionStatement(expression=expr, line=start_token.line)

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
                # 속성 접근 (점 표기)
                self.advance()
                if self.match(TokenType.IDENTIFIER):
                    attr = self.advance().value
                    expr = AttributeAccess(target=expr, attribute=attr)
                else:
                    self.error("속성 이름이 필요합니다")

            elif self.match(TokenType.POSSESSIVE):
                # 속성 접근 (소유격 '의')
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

        # 자신의 (속성 접근)
        if self.match(TokenType.SELF_ATTR):
            self.advance()
            attr_name = self.expect(TokenType.IDENTIFIER, "속성 이름이 필요합니다").value
            return SelfAccess(attribute=attr_name, line=token.line)

        # 자신 (자기 참조)
        if self.match(TokenType.SELF):
            self.advance()
            return SelfReference(line=token.line)

        # 부모의 (부모 클래스 접근)
        if self.match(TokenType.PARENT):
            self.advance()
            method_name = self.expect(TokenType.IDENTIFIER, "메서드 이름이 필요합니다").value
            args = []
            if self.match(TokenType.LPAREN):
                self.advance()
                while not self.match(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    if self.match(TokenType.COMMA):
                        self.advance()
                self.expect(TokenType.RPAREN, "')'가 필요합니다")
            return ParentCall(method=method_name, arguments=args, line=token.line)

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

    def parse_class_def(self, name: str, start_token: Token) -> ClassDef:
        """클래스(틀) 정의 파싱"""
        self.advance()  # 정의하자

        parents = []

        # 상속 확인: "동물을 확장하여" 또는 "A와 B를 확장하여"
        if self.match(TokenType.IDENTIFIER):
            # 부모 클래스 이름들 수집
            while self.match(TokenType.IDENTIFIER):
                parent_name = self.advance().value
                parents.append(parent_name)
                if self.match(TokenType.WITH):  # 과/와
                    self.advance()
                elif self.match(TokenType.OBJECT):  # 을/를
                    break

            if self.match(TokenType.OBJECT):  # 을/를
                self.advance()
            if self.match(TokenType.EXTEND):  # 확장하여
                self.advance()

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")

        body = []
        init_method = None
        destroy_method = None

        # 클래스 본문 파싱
        while not self.match(TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()

            if self.match(TokenType.DEDENT, TokenType.EOF):
                break

            # 생성자: "생성할때"
            if self.match(TokenType.INIT):
                init_method = self.parse_init_method()
                body.append(init_method)

            # 소멸자: "소멸할때"
            elif self.match(TokenType.DESTROY):
                destroy_method = self.parse_destroy_method()
                body.append(destroy_method)

            # 데코레이터: @틀메서드 또는 @정적메서드
            elif self.match(TokenType.AT):
                method = self.parse_decorated_method()
                if method:
                    body.append(method)

            # 일반 메서드: "메서드명을 정의하자"
            elif self.match(TokenType.IDENTIFIER):
                method = self.parse_method_def()
                if method:
                    body.append(method)

            else:
                # 클래스 속성 등 기타 문장
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)

            self.skip_newlines()

        if self.match(TokenType.DEDENT):
            self.advance()

        return ClassDef(
            name=name,
            parents=parents,
            body=body,
            init_method=init_method,
            destroy_method=destroy_method,
            line=start_token.line
        )

    def parse_init_method(self) -> InitMethod:
        """생성자(생성할때) 파싱"""
        token = self.advance()  # 생성할때

        parameters = []

        # 매개변수: "이름과 나이를 받아"
        if self.match(TokenType.IDENTIFIER):
            while self.match(TokenType.IDENTIFIER):
                param = self.advance().value
                parameters.append(param)
                if self.match(TokenType.WITH):  # 과/와
                    self.advance()
                elif self.match(TokenType.OBJECT):  # 을/를
                    break

            if self.match(TokenType.OBJECT):
                self.advance()
            if self.match(TokenType.RECEIVE):  # 받아
                self.advance()

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_class_body_block()

        return InitMethod(parameters=parameters, body=body, line=token.line)

    def parse_destroy_method(self) -> MethodDef:
        """소멸자(소멸할때) 파싱"""
        token = self.advance()  # 소멸할때

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_class_body_block()

        return MethodDef(name="__del__", parameters=[], body=body, line=token.line)

    def parse_method_def(self) -> Optional[MethodDef]:
        """메서드 정의 파싱"""
        start_token = self.current()
        name = self.advance().value

        # "메서드명을 정의하자"
        if not self.match(TokenType.OBJECT):
            self.pos -= 1
            return None

        self.advance()  # 을/를

        if not self.match(TokenType.DEFINE):
            self.pos -= 2
            return None

        self.advance()  # 정의하자

        parameters = []

        # 매개변수
        if self.match(TokenType.IDENTIFIER):
            while self.match(TokenType.IDENTIFIER):
                param = self.advance().value
                parameters.append(param)
                if self.match(TokenType.WITH):
                    self.advance()
                elif self.match(TokenType.OBJECT):
                    break

            if self.match(TokenType.OBJECT):
                self.advance()
            if self.match(TokenType.RECEIVE):
                self.advance()

        self.skip_newlines()
        self.expect(TokenType.INDENT, "들여쓰기가 필요합니다")
        body = self.parse_class_body_block()

        return MethodDef(
            name=name,
            parameters=parameters,
            body=body,
            line=start_token.line
        )

    def parse_decorated_method(self) -> Optional[MethodDef]:
        """데코레이터가 있는 메서드 파싱"""
        self.advance()  # @

        is_class_method = False
        is_static_method = False

        if self.match(TokenType.CLASS_METHOD):  # 틀메서드
            is_class_method = True
            self.advance()
        elif self.match(TokenType.STATIC_METHOD):  # 정적메서드
            is_static_method = True
            self.advance()

        self.skip_newlines()

        if not self.match(TokenType.IDENTIFIER):
            return None

        method = self.parse_method_def()
        if method:
            method.is_class_method = is_class_method
            method.is_static_method = is_static_method

        return method

    def parse_class_body_block(self) -> List[Statement]:
        """클래스 내부 블록 파싱 (자신의 처리 포함)"""
        statements = []
        self.skip_newlines()

        while not self.match(TokenType.DEDENT, TokenType.EOF):
            stmt = self.parse_class_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()

        if self.match(TokenType.DEDENT):
            self.advance()

        return statements

    def parse_class_statement(self) -> Optional[Statement]:
        """클래스 내부 문장 파싱"""
        self.skip_newlines()

        if self.match(TokenType.EOF, TokenType.DEDENT):
            return None

        token = self.current()

        # 부모의 생성() 패턴
        if self.match(TokenType.PARENT):  # 부모의
            return self.parse_parent_call()

        # 기타 문장 (자신의 포함 - parse_statement에서 처리)
        return self.parse_statement()

    def parse_self_assignment(self) -> Statement:
        """자신의 속성 할당 파싱"""
        start_token = self.advance()  # 자신의
        attr_name = self.expect(TokenType.IDENTIFIER, "속성 이름이 필요합니다").value

        if self.match(TokenType.TOPIC_MARKER):  # 은/는
            self.advance()
            value = self.parse_expression()
            if self.match(TokenType.IS):  # 이다
                self.advance()

            return Assignment(
                target=SelfAccess(attribute=attr_name, line=start_token.line),
                value=value,
                line=start_token.line
            )

        # 자신의 X를 출력하라 등의 경우
        self.pos -= 2  # 되돌리기
        expr = self.parse_expression()
        return ExpressionStatement(expression=expr, line=start_token.line)

    def parse_parent_call(self) -> Statement:
        """부모 클래스 호출 파싱"""
        start_token = self.advance()  # 부모의
        method_name = self.expect(TokenType.IDENTIFIER, "메서드 이름이 필요합니다").value

        args = []
        if self.match(TokenType.LPAREN):
            self.advance()
            while not self.match(TokenType.RPAREN):
                args.append(self.parse_expression())
                if self.match(TokenType.COMMA):
                    self.advance()
            self.expect(TokenType.RPAREN, "')'가 필요합니다")

        return ExpressionStatement(
            expression=ParentCall(method=method_name, arguments=args, line=start_token.line),
            line=start_token.line
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
