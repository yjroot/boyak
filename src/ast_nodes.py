"""
보약 프로그래밍 언어 AST 노드 정의
추상 구문 트리(AST)를 구성하는 노드 클래스들입니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Union


@dataclass
class ASTNode:
    """AST 노드 기본 클래스"""
    line: int = 0
    column: int = 0


# ============ 표현식 (Expression) ============

@dataclass
class Expression(ASTNode):
    """표현식 기본 클래스"""
    pass


@dataclass
class NumberLiteral(Expression):
    """숫자 리터럴"""
    value: Union[int, float] = 0


@dataclass
class StringLiteral(Expression):
    """문자열 리터럴"""
    value: str = ""


@dataclass
class BooleanLiteral(Expression):
    """불리언 리터럴"""
    value: bool = True


@dataclass
class NoneLiteral(Expression):
    """없음(None) 리터럴"""
    pass


@dataclass
class Identifier(Expression):
    """식별자 (변수명, 함수명 등)"""
    name: str = ""


@dataclass
class ListLiteral(Expression):
    """목록 리터럴"""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class DictLiteral(Expression):
    """사전 리터럴"""
    pairs: List[tuple] = field(default_factory=list)  # [(key, value), ...]


@dataclass
class TupleLiteral(Expression):
    """튜플 리터럴"""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class ListComprehension(Expression):
    """리스트 컴프리헨션: [표현식 각각의 변수를 목록에서]"""
    expression: Expression = None
    variable: str = ""
    iterable: Expression = None
    condition: Optional[Expression] = None  # 필터 조건


@dataclass
class SetLiteral(Expression):
    """집합 리터럴"""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class OptionalChain(Expression):
    """안전 속성 접근 (?.): 대상이 없음이면 없음 반환"""
    target: Expression = None
    attribute: str = ""


@dataclass
class NullCoalesce(Expression):
    """널 병합 연산자 (??): 왼쪽이 없음이면 오른쪽 반환"""
    left: Expression = None
    right: Expression = None


@dataclass
class BinaryOp(Expression):
    """이항 연산"""
    operator: str = ""  # +, -, *, /, //, %, **, ==, !=, <, >, <=, >=, and, or
    left: Expression = None
    right: Expression = None


@dataclass
class UnaryOp(Expression):
    """단항 연산"""
    operator: str = ""  # -, not
    operand: Expression = None


@dataclass
class Comparison(Expression):
    """비교 표현식 (한글 스타일)"""
    left: Expression = None
    comparator: str = ""  # 같다, 다르다, 보다크다, 보다작다, 크거나같다, 작거나같다, 에있다, 에없다
    right: Expression = None


@dataclass
class IndexAccess(Expression):
    """인덱스 접근 (목록[인덱스])"""
    target: Expression = None
    index: Expression = None


@dataclass
class SliceAccess(Expression):
    """슬라이스 접근 (목록[시작:끝])"""
    target: Expression = None
    start: Optional[Expression] = None
    end: Optional[Expression] = None
    step: Optional[Expression] = None


@dataclass
class AttributeAccess(Expression):
    """속성 접근 (객체.속성 또는 객체의 속성)"""
    target: Expression = None
    attribute: str = ""


@dataclass
class FunctionCall(Expression):
    """함수 호출"""
    function: Expression = None  # 함수 이름 또는 표현식
    arguments: List[Expression] = field(default_factory=list)
    keyword_args: dict = field(default_factory=dict)  # {이름: 값}


@dataclass
class MethodCall(Expression):
    """메서드 호출"""
    target: Expression = None
    method: str = ""
    arguments: List[Expression] = field(default_factory=list)


@dataclass
class Lambda(Expression):
    """람다 표현식"""
    parameters: List[str] = field(default_factory=list)
    body: Expression = None


@dataclass
class TernaryOp(Expression):
    """삼항 연산 (조건부 표현식)"""
    condition: Expression = None
    true_value: Expression = None
    false_value: Expression = None


@dataclass
class InterpolatedString(Expression):
    """문자열 보간"""
    parts: List[Union[str, Expression]] = field(default_factory=list)


# ============ 문장 (Statement) ============

@dataclass
class Statement(ASTNode):
    """문장 기본 클래스"""
    pass


@dataclass
class Program(ASTNode):
    """프로그램 (최상위 노드)"""
    statements: List[Statement] = field(default_factory=list)


@dataclass
class ExpressionStatement(Statement):
    """표현식 문장"""
    expression: Expression = None


@dataclass
class Assignment(Statement):
    """할당문 (변수는 값이다)"""
    target: Union[Identifier, IndexAccess, AttributeAccess] = None
    value: Expression = None
    is_constant: bool = False  # 항상 키워드 사용 시
    type_annotation: Optional[str] = None  # 타입 힌트 (예: "정수", "문자열")


@dataclass
class MultipleAssignment(Statement):
    """다중 할당문 (가, 나는 1, 2이다)"""
    targets: List[Identifier] = field(default_factory=list)
    value: Expression = None  # TupleLiteral 또는 함수 호출


@dataclass
class CompoundAssignment(Statement):
    """복합 할당문 (+=, -= 등)"""
    target: Identifier = None
    operator: str = ""  # +=, -=, *=, /=
    value: Expression = None


@dataclass
class PrintStatement(Statement):
    """출력문"""
    value: Expression = None
    newline: bool = True


@dataclass
class InputStatement(Statement):
    """입력문"""
    target: Identifier = None
    prompt: Optional[Expression] = None


@dataclass
class IfStatement(Statement):
    """조건문"""
    condition: Expression = None
    then_body: List[Statement] = field(default_factory=list)
    elif_clauses: List[tuple] = field(default_factory=list)  # [(condition, body), ...]
    else_body: Optional[List[Statement]] = None


@dataclass
class WhileStatement(Statement):
    """조건 반복문"""
    condition: Expression = None
    body: List[Statement] = field(default_factory=list)
    else_body: Optional[List[Statement]] = None  # 완료후


@dataclass
class ForRangeStatement(Statement):
    """범위 반복문 (N부터 M까지)"""
    variable: str = ""
    start: Expression = None
    end: Expression = None
    step: Optional[Expression] = None
    reverse: bool = False
    inclusive: bool = True  # 까지(True) vs 전까지(False)
    body: List[Statement] = field(default_factory=list)
    else_body: Optional[List[Statement]] = None


@dataclass
class ForEachStatement(Statement):
    """컬렉션 반복문"""
    variable: str = ""
    iterable: Expression = None
    body: List[Statement] = field(default_factory=list)
    else_body: Optional[List[Statement]] = None


@dataclass
class TimesStatement(Statement):
    """횟수 반복문 (N번 반복하자)"""
    count: Expression = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class InfiniteLoop(Statement):
    """무한 반복문 (계속 반복하자)"""
    body: List[Statement] = field(default_factory=list)


@dataclass
class BreakStatement(Statement):
    """반복 중단 (그만)"""
    pass


@dataclass
class ContinueStatement(Statement):
    """다음 반복으로 (다음으로)"""
    pass


@dataclass
class FunctionDef(Statement):
    """함수 정의"""
    name: str = ""
    parameters: List[str] = field(default_factory=list)
    default_values: dict = field(default_factory=dict)  # {매개변수: 기본값}
    body: List[Statement] = field(default_factory=list)
    varargs: bool = False  # 여러개받아
    kwargs: bool = False   # 이름붙여받아
    decorators: List[Expression] = field(default_factory=list)  # 데코레이터 목록
    param_types: dict = field(default_factory=dict)  # {매개변수: 타입} 타입 힌트
    return_type: Optional[str] = None  # 반환 타입 힌트


@dataclass
class ReturnStatement(Statement):
    """반환문"""
    value: Optional[Expression] = None


@dataclass
class TryStatement(Statement):
    """예외 처리문"""
    try_body: List[Statement] = field(default_factory=list)
    except_clauses: List[tuple] = field(default_factory=list)  # [(exception_var, body), ...]
    finally_body: Optional[List[Statement]] = None


@dataclass
class RaiseStatement(Statement):
    """예외 발생"""
    exception_type: str = ""
    message: Optional[Expression] = None


@dataclass
class ImportStatement(Statement):
    """모듈 가져오기"""
    module_name: str = ""
    alias: Optional[str] = None
    items: Optional[List[str]] = None  # from ... import 시


@dataclass
class ListAppend(Statement):
    """목록에 추가 (목록에 값을 추가하라)"""
    target: Expression = None
    value: Expression = None


@dataclass
class ListRemove(Statement):
    """목록에서 제거"""
    target: Expression = None
    value: Expression = None


@dataclass
class PassStatement(Statement):
    """아무것도 하지 않음"""
    pass


# ============ 클래스 관련 ============

@dataclass
class ClassDef(Statement):
    """클래스 정의 (틀 정의)"""
    name: str = ""
    parents: List[str] = field(default_factory=list)  # 상속받는 클래스들
    body: List[Statement] = field(default_factory=list)
    init_method: Optional['MethodDef'] = None
    destroy_method: Optional['MethodDef'] = None


@dataclass
class MethodDef(Statement):
    """메서드 정의"""
    name: str = ""
    parameters: List[str] = field(default_factory=list)
    default_values: dict = field(default_factory=dict)
    body: List[Statement] = field(default_factory=list)
    is_class_method: bool = False
    is_static_method: bool = False
    decorators: List[Expression] = field(default_factory=list)  # 데코레이터 목록


@dataclass
class InitMethod(Statement):
    """생성자 (생성할때)"""
    parameters: List[str] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)


@dataclass
class SelfAccess(Expression):
    """자기 참조 속성 접근 (자신의 X)"""
    attribute: str = ""


@dataclass
class SelfReference(Expression):
    """자기 참조 (자신)"""
    pass


@dataclass
class ParentCall(Expression):
    """부모 클래스 메서드 호출 (부모의 메서드())"""
    method: str = ""
    arguments: List[Expression] = field(default_factory=list)


# ============ 열거형 관련 ============

@dataclass
class EnumDef(Statement):
    """열거형 정의"""
    name: str = ""
    members: List[str] = field(default_factory=list)  # 멤버 이름들
    values: dict = field(default_factory=dict)  # {멤버: 값} (값이 지정된 경우)


@dataclass
class EnumAccess(Expression):
    """열거형 멤버 접근 (열거형.멤버)"""
    enum_name: str = ""
    member: str = ""


# ============ 패턴 매칭 관련 ============

@dataclass
class MatchCase:
    """패턴 매칭의 개별 케이스"""
    pattern: Expression = None  # 매칭할 패턴 (None이면 와일드카드/기본)
    guard: Optional[Expression] = None  # 조건부 패턴 (만약 조건이면)
    body: List[Statement] = field(default_factory=list)
    is_default: bool = False  # 그외에 (기본 케이스)


@dataclass
class MatchStatement(Statement):
    """패턴 매칭문 (값을 맞춰보자)"""
    subject: Expression = None  # 매칭할 대상
    cases: List[MatchCase] = field(default_factory=list)


# ============ 제너레이터 관련 ============

@dataclass
class YieldStatement(Statement):
    """yield 문 (값을 내보내라)"""
    value: Optional[Expression] = None


@dataclass
class GeneratorDef(Statement):
    """제너레이터 정의 (생성기를 정의하자)"""
    name: str = ""
    parameters: List[str] = field(default_factory=list)
    default_values: dict = field(default_factory=dict)
    body: List[Statement] = field(default_factory=list)


# ============ with 문 관련 ============

@dataclass
class WithStatement(Statement):
    """with 문 (리소스를 사용하여)"""
    context: Expression = None  # 컨텍스트 매니저 표현식
    variable: Optional[str] = None  # as 변수 (변수로)
    body: List[Statement] = field(default_factory=list)
