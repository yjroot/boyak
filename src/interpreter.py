"""
보약 프로그래밍 언어 인터프리터 (Interpreter)
AST를 실행합니다.
"""

import sys
import math
import random
from typing import Any, Dict, List, Optional, Callable
from ast_nodes import *
from parser import parse


class BreakException(Exception):
    """반복문 중단 예외"""
    pass


class ContinueException(Exception):
    """다음 반복 예외"""
    pass


class ReturnException(Exception):
    """함수 반환 예외"""
    def __init__(self, value: Any):
        self.value = value


class BoyakError(Exception):
    """보약 런타임 에러"""
    def __init__(self, message: str, line: int = 0):
        self.message = message
        self.line = line
        super().__init__(f"줄 {line}: {message}")


class Environment:
    """변수 환경 (스코프)"""

    def __init__(self, parent: Optional['Environment'] = None):
        self.variables: Dict[str, Any] = {}
        self.constants: set = set()  # 상수 목록
        self.parent = parent

    def define(self, name: str, value: Any, is_constant: bool = False):
        """변수 정의"""
        if name in self.constants:
            raise BoyakError(f"상수 '{name}'은(는) 재할당할 수 없습니다")
        self.variables[name] = value
        if is_constant:
            self.constants.add(name)

    def get(self, name: str) -> Any:
        """변수 값 가져오기"""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)
        raise BoyakError(f"정의되지 않은 변수: '{name}'")

    def set(self, name: str, value: Any):
        """변수 값 설정"""
        if name in self.constants:
            raise BoyakError(f"상수 '{name}'은(는) 재할당할 수 없습니다")
        if name in self.variables:
            self.variables[name] = value
        elif self.parent:
            self.parent.set(name, value)
        else:
            self.variables[name] = value

    def exists(self, name: str) -> bool:
        """변수 존재 확인"""
        if name in self.variables:
            return True
        if self.parent:
            return self.parent.exists(name)
        return False


class BoyakFunction:
    """보약 사용자 정의 함수"""

    def __init__(self, name: str, parameters: List[str],
                 default_values: Dict[str, Any], body: List[Statement],
                 closure: Environment):
        self.name = name
        self.parameters = parameters
        self.default_values = default_values
        self.body = body
        self.closure = closure

    def __repr__(self):
        return f"<함수 {self.name}>"


class Interpreter:
    """보약 인터프리터"""

    def __init__(self):
        self.global_env = Environment()
        self.current_env = self.global_env
        self._setup_builtins()

    def _setup_builtins(self):
        """내장 함수 설정"""
        # 타입 변환
        self.global_env.define('정수', int)
        self.global_env.define('실수', float)
        self.global_env.define('문자열', str)
        self.global_env.define('목록', list)

        # 수학 함수
        self.global_env.define('절대값', abs)
        self.global_env.define('최대값', max)
        self.global_env.define('최소값', min)
        self.global_env.define('반올림', round)
        self.global_env.define('올림', math.ceil)
        self.global_env.define('내림', math.floor)
        self.global_env.define('제곱근', math.sqrt)

        # 컬렉션 함수
        self.global_env.define('길이', len)
        self.global_env.define('합계', sum)
        self.global_env.define('정렬', sorted)
        self.global_env.define('뒤집기', lambda x: list(reversed(x)))
        self.global_env.define('범위', lambda *args: list(range(*args)))

        # 타입 확인
        self.global_env.define('타입', type)

        # 모듈
        self.global_env.define('수학', math)
        self.global_env.define('무작위', random)

    def error(self, message: str, line: int = 0):
        """런타임 에러"""
        raise BoyakError(message, line)

    def run(self, source: str) -> Any:
        """소스 코드 실행"""
        program = parse(source)
        return self.execute_program(program)

    def execute_program(self, program: Program) -> Any:
        """프로그램 실행"""
        result = None
        for statement in program.statements:
            result = self.execute(statement)
        return result

    def execute(self, node: ASTNode) -> Any:
        """노드 실행"""
        method_name = f'execute_{type(node).__name__}'
        method = getattr(self, method_name, None)
        if method:
            return method(node)
        self.error(f"실행할 수 없는 노드: {type(node).__name__}")

    def evaluate(self, node: Expression) -> Any:
        """표현식 평가"""
        method_name = f'evaluate_{type(node).__name__}'
        method = getattr(self, method_name, None)
        if method:
            return method(node)
        self.error(f"평가할 수 없는 표현식: {type(node).__name__}")

    # ============ 문장 실행 ============

    def execute_ExpressionStatement(self, node: ExpressionStatement) -> Any:
        return self.evaluate(node.expression)

    def execute_Assignment(self, node: Assignment) -> None:
        value = self.evaluate(node.value)

        if isinstance(node.target, Identifier):
            self.current_env.define(node.target.name, value, node.is_constant)
        elif isinstance(node.target, IndexAccess):
            target = self.evaluate(node.target.target)
            index = self.evaluate(node.target.index)
            target[index] = value
        elif isinstance(node.target, AttributeAccess):
            target = self.evaluate(node.target.target)
            setattr(target, node.target.attribute, value)

    def execute_CompoundAssignment(self, node: CompoundAssignment) -> None:
        current = self.current_env.get(node.target.name)
        value = self.evaluate(node.value)

        op_map = {
            '+=': lambda a, b: a + b,
            '-=': lambda a, b: a - b,
            '*=': lambda a, b: a * b,
            '/=': lambda a, b: a / b,
        }

        new_value = op_map[node.operator](current, value)
        self.current_env.set(node.target.name, new_value)

    def execute_PrintStatement(self, node: PrintStatement) -> None:
        value = self.evaluate(node.value)
        if node.newline:
            print(value)
        else:
            print(value, end='')

    def execute_InputStatement(self, node: InputStatement) -> None:
        prompt = ""
        if node.prompt:
            prompt = self.evaluate(node.prompt)
        value = input(prompt)
        self.current_env.define(node.target.name, value)

    def execute_IfStatement(self, node: IfStatement) -> Any:
        if self._is_truthy(self.evaluate(node.condition)):
            return self.execute_block(node.then_body)

        for condition, body in node.elif_clauses:
            if self._is_truthy(self.evaluate(condition)):
                return self.execute_block(body)

        if node.else_body:
            return self.execute_block(node.else_body)

    def execute_WhileStatement(self, node: WhileStatement) -> Any:
        result = None
        completed_normally = True

        try:
            while self._is_truthy(self.evaluate(node.condition)):
                try:
                    result = self.execute_block(node.body)
                except ContinueException:
                    continue
        except BreakException:
            completed_normally = False

        if completed_normally and node.else_body:
            result = self.execute_block(node.else_body)

        return result

    def execute_ForRangeStatement(self, node: ForRangeStatement) -> Any:
        start = self.evaluate(node.start)
        end = self.evaluate(node.end)
        step = self.evaluate(node.step) if node.step else 1

        if node.inclusive:
            end += 1

        if node.reverse:
            range_values = range(end - 1, start - 1, -step)
        else:
            range_values = range(start, end, step)

        result = None
        completed_normally = True

        try:
            for value in range_values:
                self.current_env.define(node.variable, value)
                try:
                    result = self.execute_block(node.body)
                except ContinueException:
                    continue
        except BreakException:
            completed_normally = False

        if completed_normally and node.else_body:
            result = self.execute_block(node.else_body)

        return result

    def execute_ForEachStatement(self, node: ForEachStatement) -> Any:
        iterable = self.evaluate(node.iterable)
        result = None
        completed_normally = True

        try:
            for item in iterable:
                self.current_env.define(node.variable, item)
                try:
                    result = self.execute_block(node.body)
                except ContinueException:
                    continue
        except BreakException:
            completed_normally = False

        if completed_normally and node.else_body:
            result = self.execute_block(node.else_body)

        return result

    def execute_TimesStatement(self, node: TimesStatement) -> Any:
        count = self.evaluate(node.count)
        result = None

        try:
            for _ in range(int(count)):
                try:
                    result = self.execute_block(node.body)
                except ContinueException:
                    continue
        except BreakException:
            pass

        return result

    def execute_InfiniteLoop(self, node: InfiniteLoop) -> Any:
        result = None

        try:
            while True:
                try:
                    result = self.execute_block(node.body)
                except ContinueException:
                    continue
        except BreakException:
            pass

        return result

    def execute_BreakStatement(self, node: BreakStatement) -> None:
        raise BreakException()

    def execute_ContinueStatement(self, node: ContinueStatement) -> None:
        raise ContinueException()

    def execute_FunctionDef(self, node: FunctionDef) -> None:
        function = BoyakFunction(
            name=node.name,
            parameters=node.parameters,
            default_values=node.default_values,
            body=node.body,
            closure=self.current_env
        )
        self.current_env.define(node.name, function)

    def execute_ReturnStatement(self, node: ReturnStatement) -> None:
        value = None
        if node.value:
            value = self.evaluate(node.value)
        raise ReturnException(value)

    def execute_TryStatement(self, node: TryStatement) -> Any:
        try:
            return self.execute_block(node.try_body)
        except BoyakError as e:
            for exc_var, exc_body in node.except_clauses:
                if exc_var:
                    self.current_env.define(exc_var, str(e))
                return self.execute_block(exc_body)
        except Exception as e:
            for exc_var, exc_body in node.except_clauses:
                if exc_var:
                    self.current_env.define(exc_var, str(e))
                return self.execute_block(exc_body)
        finally:
            if node.finally_body:
                self.execute_block(node.finally_body)

    def execute_ImportStatement(self, node: ImportStatement) -> None:
        # 간단한 내장 모듈만 지원
        modules = {
            '수학': math,
            '무작위': random,
        }
        if node.module_name in modules:
            name = node.alias or node.module_name
            self.current_env.define(name, modules[node.module_name])
        else:
            self.error(f"알 수 없는 모듈: {node.module_name}")

    def execute_block(self, statements: List[Statement]) -> Any:
        """블록 실행"""
        result = None
        for statement in statements:
            result = self.execute(statement)
        return result

    # ============ 표현식 평가 ============

    def evaluate_NumberLiteral(self, node: NumberLiteral) -> Any:
        return node.value

    def evaluate_StringLiteral(self, node: StringLiteral) -> str:
        return node.value

    def evaluate_BooleanLiteral(self, node: BooleanLiteral) -> bool:
        return node.value

    def evaluate_NoneLiteral(self, node: NoneLiteral) -> None:
        return None

    def evaluate_Identifier(self, node: Identifier) -> Any:
        return self.current_env.get(node.name)

    def evaluate_ListLiteral(self, node: ListLiteral) -> list:
        return [self.evaluate(elem) for elem in node.elements]

    def evaluate_DictLiteral(self, node: DictLiteral) -> dict:
        return {self.evaluate(k): self.evaluate(v) for k, v in node.pairs}

    def evaluate_BinaryOp(self, node: BinaryOp) -> Any:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '//': lambda a, b: a // b,
            '%': lambda a, b: a % b,
            '**': lambda a, b: a ** b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
        }

        if node.operator in ops:
            return ops[node.operator](left, right)

        self.error(f"알 수 없는 연산자: {node.operator}")

    def evaluate_UnaryOp(self, node: UnaryOp) -> Any:
        operand = self.evaluate(node.operand)

        if node.operator == '-':
            return -operand
        if node.operator == 'not':
            return not self._is_truthy(operand)

        self.error(f"알 수 없는 단항 연산자: {node.operator}")

    def evaluate_IndexAccess(self, node: IndexAccess) -> Any:
        target = self.evaluate(node.target)
        index = self.evaluate(node.index)
        return target[index]

    def evaluate_AttributeAccess(self, node: AttributeAccess) -> Any:
        target = self.evaluate(node.target)
        return getattr(target, node.attribute)

    def evaluate_FunctionCall(self, node: FunctionCall) -> Any:
        function = self.evaluate(node.function)
        args = [self.evaluate(arg) for arg in node.arguments]
        kwargs = {k: self.evaluate(v) for k, v in node.keyword_args.items()}

        # 내장 함수
        if callable(function) and not isinstance(function, BoyakFunction):
            return function(*args, **kwargs)

        # 사용자 정의 함수
        if isinstance(function, BoyakFunction):
            return self._call_function(function, args, kwargs)

        self.error(f"호출할 수 없는 객체: {function}")

    def _call_function(self, function: BoyakFunction, args: List[Any],
                       kwargs: Dict[str, Any]) -> Any:
        """사용자 정의 함수 호출"""
        # 새 환경 생성 (클로저 사용)
        env = Environment(function.closure)

        # 매개변수 바인딩
        for i, param in enumerate(function.parameters):
            if i < len(args):
                env.define(param, args[i])
            elif param in kwargs:
                env.define(param, kwargs[param])
            elif param in function.default_values:
                env.define(param, function.default_values[param])
            else:
                self.error(f"필수 매개변수 '{param}'가 없습니다")

        # 함수 실행
        prev_env = self.current_env
        self.current_env = env

        try:
            self.execute_block(function.body)
            return None
        except ReturnException as e:
            return e.value
        finally:
            self.current_env = prev_env

    def evaluate_InterpolatedString(self, node: InterpolatedString) -> str:
        """문자열 보간 평가"""
        result = ""
        for part in node.parts:
            if isinstance(part, str):
                result += part
            else:
                result += str(self.evaluate(part))
        return result

    def _is_truthy(self, value: Any) -> bool:
        """진리값 확인"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return True


def run_file(filename: str):
    """파일 실행"""
    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()

    interpreter = Interpreter()
    try:
        interpreter.run(source)
    except BoyakError as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"내부 오류: {e}", file=sys.stderr)
        sys.exit(1)


def repl():
    """대화형 인터프리터"""
    print("보약 프로그래밍 언어 v1.0")
    print("종료하려면 '종료' 또는 'exit'를 입력하세요.")
    print()

    interpreter = Interpreter()

    while True:
        try:
            line = input("보약> ")
            if line.strip() in ('종료', 'exit', 'quit'):
                break
            if not line.strip():
                continue

            result = interpreter.run(line)
            if result is not None:
                print(result)
        except BoyakError as e:
            print(f"오류: {e}")
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"오류: {e}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        repl()
