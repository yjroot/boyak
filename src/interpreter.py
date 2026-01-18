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


class BoyakMethod:
    """보약 메서드 (인스턴스에 바인딩된 함수)"""

    def __init__(self, name: str, parameters: List[str],
                 default_values: Dict[str, Any], body: List[Statement],
                 instance: 'BoyakInstance', klass: 'BoyakClass'):
        self.name = name
        self.parameters = parameters
        self.default_values = default_values
        self.body = body
        self.instance = instance
        self.klass = klass

    def __repr__(self):
        return f"<메서드 {self.klass.name}.{self.name}>"


class BoyakClass:
    """보약 클래스"""

    def __init__(self, name: str, parents: List['BoyakClass'],
                 methods: Dict[str, 'MethodDef'],
                 init_method: Optional['InitMethod'],
                 class_attrs: Dict[str, Any],
                 closure: Environment):
        self.name = name
        self.parents = parents
        self.methods = methods
        self.init_method = init_method
        self.class_attrs = class_attrs
        self.closure = closure

    def __repr__(self):
        return f"<틀 {self.name}>"

    def get_method(self, name: str) -> Optional['MethodDef']:
        """메서드 찾기 (상속 계층 탐색)"""
        if name in self.methods:
            return self.methods[name]
        for parent in self.parents:
            method = parent.get_method(name)
            if method:
                return method
        return None


class BoyakInstance:
    """보약 인스턴스 (객체)"""

    def __init__(self, klass: BoyakClass):
        self.klass = klass
        self.attributes: Dict[str, Any] = {}

    def __repr__(self):
        return f"<{self.klass.name} 객체>"

    def get(self, name: str) -> Any:
        """속성 가져오기"""
        if name in self.attributes:
            return self.attributes[name]
        # 클래스 속성 확인
        if name in self.klass.class_attrs:
            return self.klass.class_attrs[name]
        raise BoyakError(f"'{self.klass.name}' 객체에 '{name}' 속성이 없습니다")

    def set(self, name: str, value: Any):
        """속성 설정"""
        self.attributes[name] = value


class BoyakModule:
    """보약 모듈 (네임스페이스)"""

    def __init__(self, name: str, env: 'Environment'):
        self.name = name
        self.env = env

    def __repr__(self):
        return f"<모듈 {self.name}>"

    def __getattr__(self, name: str) -> Any:
        if name in ('name', 'env'):
            return object.__getattribute__(self, name)
        if self.env.exists(name):
            return self.env.get(name)
        raise AttributeError(f"모듈 '{self.name}'에 '{name}'이(가) 없습니다")


class Interpreter:
    """보약 인터프리터"""

    def __init__(self):
        self.global_env = Environment()
        self.current_env = self.global_env
        self.current_instance: Optional[BoyakInstance] = None  # 메서드 실행 시 현재 인스턴스
        self.current_class: Optional[BoyakClass] = None  # 메서드 실행 시 현재 클래스
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
            if isinstance(target, BoyakInstance):
                target.set(node.target.attribute, value)
            else:
                setattr(target, node.target.attribute, value)
        elif isinstance(node.target, SelfAccess):
            # 자신의 X = 값
            if self.current_instance is None:
                self.error("'자신의'는 메서드 내부에서만 사용할 수 있습니다", node.line)
            self.current_instance.set(node.target.attribute, value)

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
        """모듈 가져오기"""
        # 내장 모듈
        builtin_modules = {
            '수학': math,
            '무작위': random,
        }

        if node.module_name in builtin_modules:
            module = builtin_modules[node.module_name]
            if node.items:
                # from 수학 import 제곱근
                for item in node.items:
                    if hasattr(module, item):
                        self.current_env.define(item, getattr(module, item))
                    else:
                        self.error(f"'{node.module_name}'에 '{item}'이(가) 없습니다", node.line)
            else:
                name = node.alias or node.module_name
                self.current_env.define(name, module)
            return

        # 사용자 모듈 (.보약 파일)
        module_env = self._load_module(node.module_name, node.line)

        if node.items:
            # from 모듈 import 항목들
            for item in node.items:
                if module_env.exists(item):
                    self.current_env.define(item, module_env.get(item))
                else:
                    self.error(f"'{node.module_name}'에 '{item}'이(가) 없습니다", node.line)
        else:
            # 전체 모듈 가져오기
            name = node.alias or node.module_name
            # 모듈을 네임스페이스 객체로 래핑
            module_obj = BoyakModule(node.module_name, module_env)
            self.current_env.define(name, module_obj)

    def _load_module(self, module_name: str, line: int) -> Environment:
        """모듈 파일 로드 및 실행"""
        import os

        # 이미 로드된 모듈 캐시 확인
        if not hasattr(self, '_module_cache'):
            self._module_cache = {}

        if module_name in self._module_cache:
            return self._module_cache[module_name]

        # 모듈 파일 찾기
        module_paths = [
            f"{module_name}.보약",
            os.path.join(self.module_search_path, f"{module_name}.보약") if hasattr(self, 'module_search_path') else None,
        ]

        module_file = None
        for path in module_paths:
            if path and os.path.exists(path):
                module_file = path
                break

        if not module_file:
            self.error(f"모듈을 찾을 수 없습니다: '{module_name}'", line)

        # 모듈 파일 읽기 및 파싱
        with open(module_file, 'r', encoding='utf-8') as f:
            source = f.read()

        program = parse(source)

        # 새 환경에서 모듈 실행
        module_env = Environment()
        # 내장 함수 복사
        for name, value in self.global_env.variables.items():
            module_env.define(name, value)

        prev_env = self.current_env
        self.current_env = module_env

        try:
            self.execute_program(program)
        finally:
            self.current_env = prev_env

        # 캐시에 저장
        self._module_cache[module_name] = module_env

        return module_env

    def execute_ClassDef(self, node: ClassDef) -> None:
        """클래스 정의 실행"""
        # 부모 클래스 가져오기
        parents = []
        for parent_name in node.parents:
            parent = self.current_env.get(parent_name)
            if not isinstance(parent, BoyakClass):
                self.error(f"'{parent_name}'은(는) 클래스가 아닙니다", node.line)
            parents.append(parent)

        # 메서드와 클래스 속성 수집
        methods = {}
        class_attrs = {}
        init_method = None

        for stmt in node.body:
            if isinstance(stmt, MethodDef):
                methods[stmt.name] = stmt
            elif isinstance(stmt, InitMethod):
                init_method = stmt
            elif isinstance(stmt, Assignment):
                # 클래스 속성
                if isinstance(stmt.target, Identifier):
                    class_attrs[stmt.target.name] = self.evaluate(stmt.value)

        # 클래스 객체 생성
        klass = BoyakClass(
            name=node.name,
            parents=parents,
            methods=methods,
            init_method=init_method,
            class_attrs=class_attrs,
            closure=self.current_env
        )

        # 환경에 클래스 등록
        self.current_env.define(node.name, klass)

    def execute_MethodDef(self, node: MethodDef) -> None:
        """메서드 정의는 ClassDef에서 처리됨"""
        pass

    def execute_InitMethod(self, node: InitMethod) -> None:
        """생성자 정의는 ClassDef에서 처리됨"""
        pass

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

        # BoyakInstance 속성 접근
        if isinstance(target, BoyakInstance):
            # 먼저 속성 확인
            if node.attribute in target.attributes:
                return target.attributes[node.attribute]
            # 메서드 확인
            method_def = target.klass.get_method(node.attribute)
            if method_def:
                return BoyakMethod(
                    name=method_def.name,
                    parameters=method_def.parameters,
                    default_values=method_def.default_values,
                    body=method_def.body,
                    instance=target,
                    klass=target.klass
                )
            # 클래스 속성 확인
            if node.attribute in target.klass.class_attrs:
                return target.klass.class_attrs[node.attribute]
            self.error(f"'{target.klass.name}' 객체에 '{node.attribute}' 속성이 없습니다")

        # BoyakClass 속성 접근 (클래스 메서드/정적 메서드)
        if isinstance(target, BoyakClass):
            if node.attribute in target.class_attrs:
                return target.class_attrs[node.attribute]
            method_def = target.get_method(node.attribute)
            if method_def:
                if method_def.is_static_method:
                    # 정적 메서드는 인스턴스 없이 호출
                    return BoyakMethod(
                        name=method_def.name,
                        parameters=method_def.parameters,
                        default_values=method_def.default_values,
                        body=method_def.body,
                        instance=None,
                        klass=target
                    )
            self.error(f"'{target.name}' 클래스에 '{node.attribute}' 속성이 없습니다")

        return getattr(target, node.attribute)

    def evaluate_FunctionCall(self, node: FunctionCall) -> Any:
        function = self.evaluate(node.function)
        args = [self.evaluate(arg) for arg in node.arguments]
        kwargs = {k: self.evaluate(v) for k, v in node.keyword_args.items()}

        # 클래스 인스턴스 생성
        if isinstance(function, BoyakClass):
            return self._instantiate_class(function, args, kwargs)

        # 메서드 호출
        if isinstance(function, BoyakMethod):
            return self._call_method(function, args, kwargs)

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

    def _instantiate_class(self, klass: BoyakClass, args: List[Any],
                           kwargs: Dict[str, Any]) -> BoyakInstance:
        """클래스 인스턴스 생성"""
        instance = BoyakInstance(klass)

        # 생성자 호출
        if klass.init_method:
            self._call_init(klass, instance, klass.init_method, args, kwargs)
        elif args or kwargs:
            self.error(f"'{klass.name}' 클래스에는 생성자가 없습니다")

        return instance

    def _call_init(self, klass: BoyakClass, instance: BoyakInstance,
                   init_method: 'InitMethod', args: List[Any],
                   kwargs: Dict[str, Any]) -> None:
        """생성자 호출"""
        # 새 환경 생성
        env = Environment(klass.closure)

        # 매개변수 바인딩
        for i, param in enumerate(init_method.parameters):
            if i < len(args):
                env.define(param, args[i])
            elif param in kwargs:
                env.define(param, kwargs[param])
            else:
                self.error(f"생성자의 필수 매개변수 '{param}'가 없습니다")

        # 환경 및 인스턴스 설정
        prev_env = self.current_env
        prev_instance = self.current_instance
        prev_class = self.current_class

        self.current_env = env
        self.current_instance = instance
        self.current_class = klass

        try:
            self.execute_block(init_method.body)
        except ReturnException:
            pass  # 생성자에서 return은 무시
        finally:
            self.current_env = prev_env
            self.current_instance = prev_instance
            self.current_class = prev_class

    def _call_method(self, method: BoyakMethod, args: List[Any],
                     kwargs: Dict[str, Any]) -> Any:
        """메서드 호출"""
        # 정적 메서드는 인스턴스 없이 실행
        if method.instance is None:
            env = Environment(method.klass.closure)
        else:
            env = Environment(method.klass.closure)

        # 매개변수 바인딩
        for i, param in enumerate(method.parameters):
            if i < len(args):
                env.define(param, args[i])
            elif param in kwargs:
                env.define(param, kwargs[param])
            elif param in method.default_values:
                env.define(param, method.default_values[param])
            else:
                self.error(f"메서드의 필수 매개변수 '{param}'가 없습니다")

        # 환경 및 인스턴스 설정
        prev_env = self.current_env
        prev_instance = self.current_instance
        prev_class = self.current_class

        self.current_env = env
        self.current_instance = method.instance
        self.current_class = method.klass

        try:
            self.execute_block(method.body)
            return None
        except ReturnException as e:
            return e.value
        finally:
            self.current_env = prev_env
            self.current_instance = prev_instance
            self.current_class = prev_class

    def evaluate_SelfAccess(self, node: SelfAccess) -> Any:
        """자신의 X 평가"""
        if self.current_instance is None:
            self.error("'자신의'는 메서드 내부에서만 사용할 수 있습니다", node.line)
        return self.current_instance.get(node.attribute)

    def evaluate_SelfReference(self, node: SelfReference) -> BoyakInstance:
        """자신 평가"""
        if self.current_instance is None:
            self.error("'자신'은 메서드 내부에서만 사용할 수 있습니다", node.line)
        return self.current_instance

    def evaluate_ParentCall(self, node: ParentCall) -> Any:
        """부모의 메서드() 평가"""
        if self.current_instance is None or self.current_class is None:
            self.error("'부모의'는 메서드 내부에서만 사용할 수 있습니다", node.line)

        # 부모 클래스에서 메서드 찾기
        for parent in self.current_class.parents:
            method_def = parent.get_method(node.method)
            if method_def:
                # 부모 메서드를 현재 인스턴스에서 호출
                method = BoyakMethod(
                    name=method_def.name,
                    parameters=method_def.parameters,
                    default_values=method_def.default_values,
                    body=method_def.body,
                    instance=self.current_instance,
                    klass=parent
                )
                args = [self.evaluate(arg) for arg in node.arguments]
                return self._call_method(method, args, {})

        # 특수 케이스: 부모의 생성자 호출
        if node.method == '생성':
            for parent in self.current_class.parents:
                if parent.init_method:
                    args = [self.evaluate(arg) for arg in node.arguments]
                    self._call_init(parent, self.current_instance, parent.init_method, args, {})
                    return None

        self.error(f"부모 클래스에 '{node.method}' 메서드가 없습니다", node.line)

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
    import os

    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()

    interpreter = Interpreter()
    # 실행 파일의 디렉토리를 모듈 검색 경로로 설정
    interpreter.module_search_path = os.path.dirname(os.path.abspath(filename))

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
