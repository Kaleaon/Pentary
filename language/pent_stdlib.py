#!/usr/bin/env python3
"""
Pent Standard Library
Provides built-in functions and types for the Pent programming language
"""

from typing import Dict, List, Any, Callable, Tuple
import math


class PentStdLib:
    """Standard library for Pent language runtime"""
    
    # Built-in function signatures: name -> (param_types, return_type, implementation)
    FUNCTIONS: Dict[str, Tuple[List[str], str, Callable]] = {}
    
    # Built-in types
    TYPES = {
        'pent': {'size': 16, 'range': (-76293945312, 76293945312)},
        'int': {'size': 32, 'range': (-2147483648, 2147483647)},
        'float': {'size': 32, 'precision': 6},
        'bool': {'size': 1, 'values': [True, False]},
        'void': {'size': 0},
        'string': {'size': 'variable'},
    }
    
    @classmethod
    def register_function(cls, name: str, param_types: List[str], return_type: str, 
                         impl: Callable):
        """Register a built-in function"""
        cls.FUNCTIONS[name] = (param_types, return_type, impl)
    
    @classmethod
    def call(cls, name: str, *args) -> Any:
        """Call a built-in function"""
        if name not in cls.FUNCTIONS:
            raise ValueError(f"Unknown function: {name}")
        
        param_types, return_type, impl = cls.FUNCTIONS[name]
        return impl(*args)


# ============================================================================
# I/O Functions
# ============================================================================

def _print(*args):
    """Print without newline"""
    print(*args, end='')
    return None

def _println(*args):
    """Print with newline"""
    print(*args)
    return None

def _read():
    """Read line from input"""
    return input()

def _read_int():
    """Read integer from input"""
    return int(input())

def _read_pent():
    """Read pentary number from input"""
    s = input()
    # Try to parse as decimal first
    try:
        return int(s)
    except ValueError:
        # Assume it's pentary literal
        from pentary_converter import PentaryConverter
        return PentaryConverter.pentary_to_decimal(s)


PentStdLib.register_function('print', ['any'], 'void', _print)
PentStdLib.register_function('println', ['any'], 'void', _println)
PentStdLib.register_function('read', [], 'string', _read)
PentStdLib.register_function('read_int', [], 'int', _read_int)
PentStdLib.register_function('read_pent', [], 'pent', _read_pent)


# ============================================================================
# Math Functions
# ============================================================================

def _abs(x):
    """Absolute value"""
    return abs(x)

def _min(a, b):
    """Minimum of two values"""
    return min(a, b)

def _max(a, b):
    """Maximum of two values"""
    return max(a, b)

def _clamp(x, min_val, max_val):
    """Clamp value to range [min_val, max_val]"""
    return max(min_val, min(x, max_val))

def _sign(x):
    """Sign of number: -1, 0, or 1"""
    if x < 0:
        return -1
    elif x > 0:
        return 1
    return 0

def _pow(base, exp):
    """Power function"""
    return int(base ** exp)

def _sqrt(x):
    """Square root (integer, truncated)"""
    return int(math.sqrt(x))

def _log2(x):
    """Base-2 logarithm (integer, truncated)"""
    if x <= 0:
        raise ValueError("log2 of non-positive number")
    return int(math.log2(x))

def _log5(x):
    """Base-5 logarithm (integer, truncated) - useful for pentary"""
    if x <= 0:
        raise ValueError("log5 of non-positive number")
    return int(math.log(x) / math.log(5))


PentStdLib.register_function('abs', ['pent'], 'pent', _abs)
PentStdLib.register_function('min', ['pent', 'pent'], 'pent', _min)
PentStdLib.register_function('max', ['pent', 'pent'], 'pent', _max)
PentStdLib.register_function('clamp', ['pent', 'pent', 'pent'], 'pent', _clamp)
PentStdLib.register_function('sign', ['pent'], 'pent', _sign)
PentStdLib.register_function('pow', ['pent', 'pent'], 'pent', _pow)
PentStdLib.register_function('sqrt', ['pent'], 'pent', _sqrt)
PentStdLib.register_function('log2', ['pent'], 'pent', _log2)
PentStdLib.register_function('log5', ['pent'], 'pent', _log5)


# ============================================================================
# Pentary-Specific Functions
# ============================================================================

def _to_pentary(decimal_value):
    """Convert decimal to pentary string representation"""
    from pentary_converter import PentaryConverter
    return PentaryConverter.decimal_to_pentary(decimal_value)

def _from_pentary(pentary_str):
    """Convert pentary string to decimal"""
    from pentary_converter import PentaryConverter
    return PentaryConverter.pentary_to_decimal(pentary_str)

def _pent_digits(value):
    """Get number of pentary digits in a value"""
    if value == 0:
        return 1
    return int(math.log(abs(value)) / math.log(5)) + 1

def _pent_digit(value, position):
    """Get pentary digit at position (0 = least significant)"""
    from pentary_converter import PentaryConverter
    pentary = PentaryConverter.decimal_to_pentary(value)
    if position >= len(pentary):
        return 0
    return PentaryConverter.SYMBOL_TO_VALUE.get(pentary[-(position + 1)], 0)

def _quantize_5(value, scale=1.0):
    """Quantize to 5 levels {-2, -1, 0, 1, 2}"""
    scaled = value / scale
    if scaled < -1.5:
        return -2
    elif scaled < -0.5:
        return -1
    elif scaled < 0.5:
        return 0
    elif scaled < 1.5:
        return 1
    else:
        return 2


PentStdLib.register_function('to_pentary', ['int'], 'string', _to_pentary)
PentStdLib.register_function('from_pentary', ['string'], 'pent', _from_pentary)
PentStdLib.register_function('pent_digits', ['pent'], 'int', _pent_digits)
PentStdLib.register_function('pent_digit', ['pent', 'int'], 'pent', _pent_digit)
PentStdLib.register_function('quantize_5', ['float'], 'pent', _quantize_5)


# ============================================================================
# Neural Network Functions
# ============================================================================

def _relu(x):
    """ReLU activation function"""
    return max(0, x)

def _leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return x if x > 0 else alpha * x

def _sigmoid_approx(x):
    """
    Approximate sigmoid for integer arithmetic.
    Piecewise linear approximation mapping [-4, 4] to [0, 250].
    
    Constants:
    - INPUT_RANGE: 8 (from -4 to 4)
    - OUTPUT_RANGE: 250 (scaled output)
    - SCALE_FACTOR: 250 / 8 = 31.25
    """
    INPUT_MIN = -4
    INPUT_MAX = 4
    OUTPUT_MIN = 0
    OUTPUT_MAX = 250
    
    if x < INPUT_MIN:
        return OUTPUT_MIN
    elif x > INPUT_MAX:
        return OUTPUT_MAX
    else:
        # Linear interpolation: maps [INPUT_MIN, INPUT_MAX] to [OUTPUT_MIN, OUTPUT_MAX]
        return int((x - INPUT_MIN) * OUTPUT_MAX / (INPUT_MAX - INPUT_MIN))

def _tanh_approx(x):
    """Approximate tanh for integer arithmetic"""
    # Piecewise linear approximation
    if x < -4:
        return -1
    elif x > 4:
        return 1
    else:
        return x / 4  # Linear approximation in [-4, 4]


PentStdLib.register_function('relu', ['pent'], 'pent', _relu)
PentStdLib.register_function('leaky_relu', ['pent', 'float'], 'pent', _leaky_relu)
PentStdLib.register_function('sigmoid_approx', ['pent'], 'pent', _sigmoid_approx)
PentStdLib.register_function('tanh_approx', ['pent'], 'pent', _tanh_approx)


# ============================================================================
# Array Functions
# ============================================================================

def _array_new(size, init_value=0):
    """Create new array with given size and initial value"""
    return [init_value] * size

def _array_len(arr):
    """Get array length"""
    return len(arr)

def _array_get(arr, index):
    """Get element at index"""
    if index < 0 or index >= len(arr):
        raise IndexError(f"Array index {index} out of bounds")
    return arr[index]

def _array_set(arr, index, value):
    """Set element at index"""
    if index < 0 or index >= len(arr):
        raise IndexError(f"Array index {index} out of bounds")
    arr[index] = value
    return None

def _array_sum(arr):
    """Sum of all elements"""
    return sum(arr)

def _array_max(arr):
    """Maximum element"""
    if not arr:
        raise ValueError("Cannot find max of empty array")
    return max(arr)

def _array_min(arr):
    """Minimum element"""
    if not arr:
        raise ValueError("Cannot find min of empty array")
    return min(arr)

def _array_dot(a, b):
    """Dot product of two arrays"""
    if len(a) != len(b):
        raise ValueError("Arrays must have same length for dot product")
    return sum(x * y for x, y in zip(a, b))


PentStdLib.register_function('array_new', ['int', 'pent'], 'array', _array_new)
PentStdLib.register_function('array_len', ['array'], 'int', _array_len)
PentStdLib.register_function('array_get', ['array', 'int'], 'pent', _array_get)
PentStdLib.register_function('array_set', ['array', 'int', 'pent'], 'void', _array_set)
PentStdLib.register_function('array_sum', ['array'], 'pent', _array_sum)
PentStdLib.register_function('array_max', ['array'], 'pent', _array_max)
PentStdLib.register_function('array_min', ['array'], 'pent', _array_min)
PentStdLib.register_function('array_dot', ['array', 'array'], 'pent', _array_dot)


# ============================================================================
# Matrix Functions
# ============================================================================

def _matrix_new(rows, cols, init_value=0):
    """Create new matrix with given dimensions"""
    return [[init_value] * cols for _ in range(rows)]

def _matrix_rows(matrix):
    """Get number of rows"""
    return len(matrix)

def _matrix_cols(matrix):
    """Get number of columns"""
    if not matrix:
        return 0
    return len(matrix[0])

def _matrix_get(matrix, row, col):
    """Get element at (row, col)"""
    return matrix[row][col]

def _matrix_set(matrix, row, col, value):
    """Set element at (row, col)"""
    matrix[row][col] = value
    return None

def _matrix_multiply(a, b):
    """Matrix multiplication (a @ b)"""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    
    if cols_a != rows_b:
        raise ValueError(f"Matrix dimensions incompatible: {cols_a} != {rows_b}")
    
    result = [[0] * cols_b for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    
    return result

def _matrix_vector_multiply(matrix, vector):
    """Matrix-vector multiplication"""
    rows = len(matrix)
    cols = len(matrix[0])
    
    if cols != len(vector):
        raise ValueError(f"Dimensions incompatible: {cols} != {len(vector)}")
    
    result = [0] * rows
    for i in range(rows):
        for j in range(cols):
            result[i] += matrix[i][j] * vector[j]
    
    return result

def _matrix_transpose(matrix):
    """Transpose matrix"""
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


PentStdLib.register_function('matrix_new', ['int', 'int', 'pent'], 'matrix', _matrix_new)
PentStdLib.register_function('matrix_rows', ['matrix'], 'int', _matrix_rows)
PentStdLib.register_function('matrix_cols', ['matrix'], 'int', _matrix_cols)
PentStdLib.register_function('matrix_get', ['matrix', 'int', 'int'], 'pent', _matrix_get)
PentStdLib.register_function('matrix_set', ['matrix', 'int', 'int', 'pent'], 'void', _matrix_set)
PentStdLib.register_function('matrix_multiply', ['matrix', 'matrix'], 'matrix', _matrix_multiply)
PentStdLib.register_function('matvec', ['matrix', 'array'], 'array', _matrix_vector_multiply)
PentStdLib.register_function('matrix_transpose', ['matrix'], 'matrix', _matrix_transpose)


# ============================================================================
# String Functions
# ============================================================================

def _strlen(s):
    """String length"""
    return len(s)

def _str_concat(a, b):
    """Concatenate strings"""
    return a + b

def _str_substr(s, start, length):
    """Get substring"""
    return s[start:start + length]

def _str_char_at(s, index):
    """Get character at index"""
    return s[index]

def _str_to_int(s):
    """Parse string as integer"""
    return int(s)

def _int_to_str(n):
    """Convert integer to string"""
    return str(n)


PentStdLib.register_function('strlen', ['string'], 'int', _strlen)
PentStdLib.register_function('str_concat', ['string', 'string'], 'string', _str_concat)
PentStdLib.register_function('str_substr', ['string', 'int', 'int'], 'string', _str_substr)
PentStdLib.register_function('str_char_at', ['string', 'int'], 'string', _str_char_at)
PentStdLib.register_function('str_to_int', ['string'], 'int', _str_to_int)
PentStdLib.register_function('int_to_str', ['int'], 'string', _int_to_str)


# ============================================================================
# Type Checking Functions
# ============================================================================

def _is_zero(x):
    """Check if value is zero"""
    return x == 0

def _is_positive(x):
    """Check if value is positive"""
    return x > 0

def _is_negative(x):
    """Check if value is negative"""
    return x < 0

def _is_even(x):
    """Check if value is even (in decimal)"""
    return x % 2 == 0

def _is_pentary_even(x):
    """Check if value is 'even' in pentary (divisible by 5)"""
    return x % 5 == 0


PentStdLib.register_function('is_zero', ['pent'], 'bool', _is_zero)
PentStdLib.register_function('is_positive', ['pent'], 'bool', _is_positive)
PentStdLib.register_function('is_negative', ['pent'], 'bool', _is_negative)
PentStdLib.register_function('is_even', ['pent'], 'bool', _is_even)
PentStdLib.register_function('is_pentary_even', ['pent'], 'bool', _is_pentary_even)


# ============================================================================
# Bitwise-like Operations (Pentary-adapted)
# ============================================================================

def _pent_and(a, b):
    """Pentary AND: min of each digit position"""
    from pentary_converter import PentaryConverter
    a_pent = PentaryConverter.decimal_to_pentary(a)
    b_pent = PentaryConverter.decimal_to_pentary(b)
    
    # Pad to same length using proper pentary zero padding
    max_len = max(len(a_pent), len(b_pent))
    while len(a_pent) < max_len:
        a_pent = '0' + a_pent  # Prepend pentary zero (not ASCII '0' padding)
    while len(b_pent) < max_len:
        b_pent = '0' + b_pent
    
    result = ''
    for i in range(max_len):
        a_val = PentaryConverter.SYMBOL_TO_VALUE.get(a_pent[i], 0)
        b_val = PentaryConverter.SYMBOL_TO_VALUE.get(b_pent[i], 0)
        result += PentaryConverter.VALUE_TO_SYMBOL[min(a_val, b_val)]
    
    return PentaryConverter.pentary_to_decimal(result)

def _pent_or(a, b):
    """Pentary OR: max of each digit position"""
    from pentary_converter import PentaryConverter
    a_pent = PentaryConverter.decimal_to_pentary(a)
    b_pent = PentaryConverter.decimal_to_pentary(b)
    
    # Pad to same length using proper pentary zero padding
    max_len = max(len(a_pent), len(b_pent))
    while len(a_pent) < max_len:
        a_pent = '0' + a_pent
    while len(b_pent) < max_len:
        b_pent = '0' + b_pent
    
    result = ''
    for i in range(max_len):
        a_val = PentaryConverter.SYMBOL_TO_VALUE.get(a_pent[i], 0)
        b_val = PentaryConverter.SYMBOL_TO_VALUE.get(b_pent[i], 0)
        result += PentaryConverter.VALUE_TO_SYMBOL[max(a_val, b_val)]
    
    return PentaryConverter.pentary_to_decimal(result)

def _pent_not(a):
    """Pentary NOT: negate each digit"""
    from pentary_converter import PentaryConverter
    return PentaryConverter.pentary_to_decimal(
        PentaryConverter.negate_pentary(PentaryConverter.decimal_to_pentary(a))
    )


PentStdLib.register_function('pent_and', ['pent', 'pent'], 'pent', _pent_and)
PentStdLib.register_function('pent_or', ['pent', 'pent'], 'pent', _pent_or)
PentStdLib.register_function('pent_not', ['pent'], 'pent', _pent_not)


# ============================================================================
# Memory Management (for runtime)
# ============================================================================

class PentHeap:
    """Simple heap implementation for Pent runtime"""
    
    def __init__(self, size: int = 65536):
        self.memory = [0] * size
        self.free_list = [(0, size)]  # (start, length) pairs
        self.allocated = {}  # ptr -> length
    
    def alloc(self, size: int) -> int:
        """Allocate memory, return pointer (address)"""
        for i, (start, length) in enumerate(self.free_list):
            if length >= size:
                # Allocate from this block
                ptr = start
                self.allocated[ptr] = size
                
                if length > size:
                    # Split block
                    self.free_list[i] = (start + size, length - size)
                else:
                    # Exact fit, remove block
                    self.free_list.pop(i)
                
                return ptr
        
        raise MemoryError("Out of memory")
    
    def free(self, ptr: int):
        """Free allocated memory"""
        if ptr not in self.allocated:
            raise ValueError(f"Invalid pointer: {ptr}")
        
        size = self.allocated.pop(ptr)
        
        # Add to free list
        self.free_list.append((ptr, size))
        
        # Coalesce adjacent free blocks
        self.free_list.sort()
        coalesced = []
        for start, length in self.free_list:
            if coalesced and coalesced[-1][0] + coalesced[-1][1] == start:
                # Merge with previous
                coalesced[-1] = (coalesced[-1][0], coalesced[-1][1] + length)
            else:
                coalesced.append((start, length))
        self.free_list = coalesced
    
    def read(self, ptr: int, offset: int = 0) -> int:
        """Read value at address"""
        addr = ptr + offset
        if addr < 0 or addr >= len(self.memory):
            raise MemoryError(f"Invalid address: {addr}")
        return self.memory[addr]
    
    def write(self, ptr: int, offset: int, value: int):
        """Write value at address"""
        addr = ptr + offset
        if addr < 0 or addr >= len(self.memory):
            raise MemoryError(f"Invalid address: {addr}")
        self.memory[addr] = value


# Global heap for runtime
_pent_heap = PentHeap()

def _alloc(size):
    """Allocate memory"""
    return _pent_heap.alloc(size)

def _free(ptr):
    """Free memory"""
    _pent_heap.free(ptr)
    return None

def _load(ptr, offset=0):
    """Load from memory"""
    return _pent_heap.read(ptr, offset)

def _store(ptr, offset, value):
    """Store to memory"""
    _pent_heap.write(ptr, offset, value)
    return None


PentStdLib.register_function('alloc', ['int'], 'int', _alloc)
PentStdLib.register_function('free', ['int'], 'void', _free)
PentStdLib.register_function('load', ['int', 'int'], 'pent', _load)
PentStdLib.register_function('store', ['int', 'int', 'pent'], 'void', _store)


# ============================================================================
# Test Suite
# ============================================================================

def test_stdlib():
    """Test standard library functions"""
    print("Testing Pent Standard Library...")
    
    # Test math functions
    assert PentStdLib.call('abs', -5) == 5
    assert PentStdLib.call('min', 3, 7) == 3
    assert PentStdLib.call('max', 3, 7) == 7
    assert PentStdLib.call('clamp', 10, 0, 5) == 5
    assert PentStdLib.call('sign', -10) == -1
    assert PentStdLib.call('pow', 5, 3) == 125
    
    # Test pentary functions
    assert PentStdLib.call('relu', -5) == 0
    assert PentStdLib.call('relu', 5) == 5
    
    # Test array functions
    arr = PentStdLib.call('array_new', 5, 0)
    assert PentStdLib.call('array_len', arr) == 5
    PentStdLib.call('array_set', arr, 2, 42)
    assert PentStdLib.call('array_get', arr, 2) == 42
    
    # Test matrix functions
    mat = PentStdLib.call('matrix_new', 2, 3, 1)
    assert PentStdLib.call('matrix_rows', mat) == 2
    assert PentStdLib.call('matrix_cols', mat) == 3
    
    # Test memory functions
    ptr = PentStdLib.call('alloc', 10)
    PentStdLib.call('store', ptr, 0, 42)
    assert PentStdLib.call('load', ptr, 0) == 42
    PentStdLib.call('free', ptr)
    
    print("All standard library tests passed!")


if __name__ == '__main__':
    test_stdlib()
