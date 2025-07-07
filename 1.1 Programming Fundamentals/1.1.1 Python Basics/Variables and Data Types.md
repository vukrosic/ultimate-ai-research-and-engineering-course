# Python Fundamentals: Variables and Data Types

## Part 1: Your First Variables

```python
# Creating your first variables
name = "Alice"
age = 25
height = 5.6
is_student = True

print(f"Name: {name}")
print(f"Age: {age}")
print(f"Height: {height} feet")
print(f"Is student: {is_student}")
```

**What happened**: We created four different variables to store different types of information. Python automatically figured out what type each variable should be.

## Part 2: Understanding Data Types

#### Basic Data Types

```python
# The four fundamental data types
number = 42          # Integer (whole numbers)
price = 19.99        # Float (decimal numbers)
message = "Hello!"   # String (text)
ready = False        # Boolean (True or False)

# Check what type each variable is
print(f"number is type: {type(number)}")
print(f"price is type: {type(price)}")
print(f"message is type: {type(message)}")
print(f"ready is type: {type(ready)}")
```

#### OPTIONAL: Understanding the type() Function

```python
# Let's explore type() in more detail
print("Exploring type() function:")
print(f"type(42) = {type(42)}")           # <class 'int'>
print(f"type(3.14) = {type(3.14)}")       # <class 'float'>
print(f"type('hello') = {type('hello')}")  # <class 'str'>
print(f"type(True) = {type(True)}")       # <class 'bool'>

# Type names without the <class '...'> wrapper
print(f"\nJust the type names:")
print(f"type(42).__name__ = {type(42).__name__}")
print(f"type(3.14).__name__ = {type(3.14).__name__}")
```

#### Working with Integers

```python
# Integers - whole numbers
students_count = 30
temperature = -5
big_number = 1000000

print(f"Students: {students_count}")
print(f"Temperature: {temperature}°C")
print(f"Big number: {big_number}")

# Basic math with integers
result = students_count + 5
print(f"After adding 5 more students: {result}")
```

#### OPTIONAL: Integer Operations and Limits

```python
# Python integers can be very large (unlike some other languages)
huge_number = 10 ** 100  # 10 to the power of 100
print(f"Huge number: {huge_number}")
print(f"Length of huge number: {len(str(huge_number))} digits")

# Common integer operations
a = 17
b = 5
print(f"\nInteger operations with {a} and {b}:")
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")        # Always returns float
print(f"Floor division: {a} // {b} = {a // b}")  # Integer division
print(f"Remainder: {a} % {b} = {a % b}")       # Modulo operation
print(f"Power: {a} ** {b} = {a ** b}")         # Exponentiation
```

#### Working with Floats

```python
# Floats - decimal numbers
weight = 68.5
pi = 3.14159
bank_balance = 1250.75

print(f"Weight: {weight} kg")
print(f"Pi: {pi}")
print(f"Bank balance: ${bank_balance}")

# Floats from integer division
result = 10 / 3
print(f"10 divided by 3: {result}")
```

#### OPTIONAL: Float Precision and Formatting

```python
# Floats have limited precision
print("Float precision examples:")
result = 0.1 + 0.2
print(f"0.1 + 0.2 = {result}")  # Might not be exactly 0.3!
print(f"Is 0.1 + 0.2 == 0.3? {result == 0.3}")

# Formatting floats for display
value = 3.14159265359
print(f"\nFormatting {value}:")
print(f"2 decimal places: {value:.2f}")
print(f"4 decimal places: {value:.4f}")
print(f"Scientific notation: {value:.2e}")

# Rounding
print(f"\nRounding examples:")
print(f"round(3.7) = {round(3.7)}")
print(f"round(3.14159, 2) = {round(3.14159, 2)}")
```

#### Working with Strings

```python
# Strings - text data
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name  # String concatenation

print(f"First name: {first_name}")
print(f"Last name: {last_name}")
print(f"Full name: {full_name}")

# Different ways to create strings
single_quotes = 'Hello'
double_quotes = "Hello"
triple_quotes = """Hello"""

print(f"All these are strings: {single_quotes}, {double_quotes}, {triple_quotes}")
```

#### OPTIONAL: String Quotes and Escaping

```python
# When to use different quote types
sentence1 = "She said 'Hello there!'"  # Double quotes to contain single quotes
sentence2 = 'He replied "How are you?"'  # Single quotes to contain double quotes
print(sentence1)
print(sentence2)

# Escape characters
escaped = "She said \"Hello\" and he said 'Hi'"  # Escaping quotes with backslash
newline = "First line\nSecond line"              # \n for new line
tab = "Name:\tJohn"                              # \t for tab
print(escaped)
print(newline)
print(tab)

# Raw strings (no escaping)
file_path = r"C:\Users\John\Documents"  # r prefix means raw string
print(f"File path: {file_path}")
```

#### OPTIONAL: String Methods and Operations

```python
text = "Python Programming"

print(f"Original: '{text}'")
print(f"Length: {len(text)} characters")
print(f"Uppercase: '{text.upper()}'")
print(f"Lowercase: '{text.lower()}'")
print(f"Title case: '{text.title()}'")

# String checking methods
print(f"\nString checks:")
print(f"Starts with 'Python': {text.startswith('Python')}")
print(f"Ends with 'ing': {text.endswith('ing')}")
print(f"Contains 'gram': {'gram' in text}")

# String manipulation
print(f"\nString manipulation:")
print(f"Replace 'Python' with 'Java': '{text.replace('Python', 'Java')}'")
print(f"Split into words: {text.split()}")
print(f"Split on 'o': {text.split('o')}")
```

#### Working with Booleans

```python
# Booleans - True or False
is_sunny = True
is_raining = False
has_umbrella = True

print(f"Is sunny: {is_sunny}")
print(f"Is raining: {is_raining}")
print(f"Has umbrella: {has_umbrella}")

# Boolean logic
should_go_outside = is_sunny and not is_raining
need_umbrella = is_raining and not has_umbrella

print(f"Should go outside: {should_go_outside}")
print(f"Need umbrella: {need_umbrella}")
```

#### OPTIONAL: Boolean Operations and Truthiness

```python
# Boolean operators
a = True
b = False

print("Boolean operations:")
print(f"a = {a}, b = {b}")
print(f"a and b = {a and b}")  # Both must be True
print(f"a or b = {a or b}")    # At least one must be True
print(f"not a = {not a}")      # Opposite of a
print(f"not b = {not b}")      # Opposite of b

# Comparison operators create booleans
x = 10
y = 5
print(f"\nComparison operations:")
print(f"{x} > {y} = {x > y}")
print(f"{x} < {y} = {x < y}")
print(f"{x} == {y} = {x == y}")  # Equal to
print(f"{x} != {y} = {x != y}")  # Not equal to
```

#### OPTIONAL: Truthiness in Python

```python
# In Python, many values can be treated as True or False
print("Truthiness examples:")

# Falsy values (evaluate to False)
falsy_values = [False, 0, 0.0, "", [], {}, None]
for value in falsy_values:
    print(f"bool({repr(value)}) = {bool(value)}")

print()

# Truthy values (evaluate to True)
truthy_values = [True, 1, -1, "hello", [1, 2], {"a": 1}]
for value in truthy_values:
    print(f"bool({repr(value)}) = {bool(value)}")
```

## Part 3: Type Conversion

#### Converting Between Types

```python
# Starting with different types
number_as_string = "42"
decimal_as_string = "3.14"
age = 25
price = 19.99

print("Original values and types:")
print(f"'{number_as_string}' is {type(number_as_string).__name__}")
print(f"'{decimal_as_string}' is {type(decimal_as_string).__name__}")
print(f"{age} is {type(age).__name__}")
print(f"{price} is {type(price).__name__}")

# Convert strings to numbers
converted_int = int(number_as_string)      # String to integer
converted_float = float(decimal_as_string)  # String to float
age_as_string = str(age)                   # Integer to string
price_as_int = int(price)                  # Float to integer (loses decimal)

print("\nAfter conversion:")
print(f"{converted_int} is {type(converted_int).__name__}")
print(f"{converted_float} is {type(converted_float).__name__}")
print(f"'{age_as_string}' is {type(age_as_string).__name__}")
print(f"{price_as_int} is {type(price_as_int).__name__}")
```

#### OPTIONAL: Safe Type Conversion

```python
# Sometimes conversion can fail
test_values = ["123", "12.34", "hello", ""]

print("Testing conversions:")
for value in test_values:
    print(f"\nTesting: '{value}'")
    
    # Try converting to int
    try:
        result = int(value)
        print(f"  int('{value}') = {result}")
    except ValueError as e:
        print(f"  int('{value}') failed: {e}")
    
    # Try converting to float
    try:
        result = float(value)
        print(f"  float('{value}') = {result}")
    except ValueError as e:
        print(f"  float('{value}') failed: {e}")
```

#### OPTIONAL: Checking Before Converting

```python
# Check if a string can be converted to a number
def is_number(text):
    """Check if a string represents a number"""
    try:
        float(text)
        return True
    except ValueError:
        return False

def is_integer(text):
    """Check if a string represents an integer"""
    try:
        int(text)
        return True
    except ValueError:
        return False

# Test the functions
test_strings = ["123", "12.34", "hello", "-5", "0"]

for s in test_strings:
    print(f"'{s}': is_number={is_number(s)}, is_integer={is_integer(s)}")
```

## Part 4: Variables and Memory

#### Variable Assignment and References

```python
# Creating variables
original = 42
copy = original  # This creates a new reference to the same value

print(f"original = {original}")
print(f"copy = {copy}")

# For numbers, strings, and booleans, changing one doesn't affect the other
original = 100
print(f"After changing original:")
print(f"original = {original}")
print(f"copy = {copy}")  # copy is unchanged
```

#### OPTIONAL: Understanding Variable Assignment

```python
# Variables are like labels pointing to objects in memory
print("Understanding variable assignment:")

# Multiple variables can point to the same object
a = 1000
b = 1000
print(f"a = {a}, b = {b}")
print(f"a is b: {a is b}")  # This might be False for large numbers
print(f"id(a): {id(a)}")    # Memory address of a
print(f"id(b): {id(b)}")    # Memory address of b

# Small integers are cached by Python
x = 5
y = 5
print(f"\nSmall integers:")
print(f"x = {x}, y = {y}")
print(f"x is y: {x is y}")  # This will be True
print(f"id(x): {id(x)}")
print(f"id(y): {id(y)}")
```

#### Variable Naming Rules

```python
# Valid variable names
student_count = 30      # Snake case (recommended)
studentCount = 30       # Camel case (less common in Python)
student2 = 25          # Numbers at the end are OK
_private = "secret"     # Starting with underscore (has special meaning)

print(f"All these variables work: {student_count}, {studentCount}, {student2}")

# Python naming conventions
first_name = "John"     # Use snake_case for variables
CONSTANT_VALUE = 100    # Use UPPER_CASE for constants
ClassName = "Example"   # Use CamelCase for classes (we'll learn about these later)
```

#### OPTIONAL: Variable Naming Rules and Conventions

```python
# These would cause errors (don't run these lines):
# 2students = 30        # Can't start with number
# class = "Math"        # Can't use Python keywords
# first-name = "John"   # Can't use hyphens

# Check if a name is a valid identifier
import keyword

test_names = ["student_count", "2students", "class", "first-name", "hello_world"]

for name in test_names:
    is_valid = name.isidentifier() and not keyword.iskeyword(name)
    issues = []
    
    if not name.isidentifier():
        issues.append("invalid identifier")
    if keyword.iskeyword(name):
        issues.append("Python keyword")
    
    status = "✓ Valid" if is_valid else f"✗ Invalid ({', '.join(issues)})"
    print(f"'{name}': {status}")

# See all Python keywords
print(f"\nPython keywords: {keyword.kwlist}")
```

## Part 5: Working with Collections

#### Lists - Ordered Collections

```python
# Creating lists
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = ["hello", 42, 3.14, True]  # Lists can hold different types

print(f"Fruits: {fruits}")
print(f"Numbers: {numbers}")
print(f"Mixed types: {mixed}")

# Accessing list elements
print(f"First fruit: {fruits[0]}")      # First element (index 0)
print(f"Last fruit: {fruits[-1]}")      # Last element (negative indexing)
print(f"Second number: {numbers[1]}")   # Second element (index 1)
```

#### OPTIONAL: List Indexing and Slicing

```python
# List indexing examples
colors = ["red", "green", "blue", "yellow", "purple"]
print(f"Colors: {colors}")

# Positive indexing (from the start)
print(f"Index 0: {colors[0]}")
print(f"Index 2: {colors[2]}")

# Negative indexing (from the end)
print(f"Index -1: {colors[-1]}")  # Last element
print(f"Index -2: {colors[-2]}")  # Second to last

# Slicing (getting multiple elements)
print(f"First 3 colors: {colors[0:3]}")    # Elements 0, 1, 2
print(f"Colors[1:4]: {colors[1:4]}")       # Elements 1, 2, 3
print(f"Last 2 colors: {colors[-2:]}")     # Last 2 elements
print(f"All but first: {colors[1:]}")      # From index 1 to end
print(f"All but last: {colors[:-1]}")      # From start to second-to-last
```

#### OPTIONAL: Modifying Lists

```python
# Lists are mutable (can be changed)
shopping = ["milk", "bread", "eggs"]
print(f"Original list: {shopping}")

# Add items
shopping.append("cheese")              # Add to end
print(f"After append: {shopping}")

shopping.insert(1, "butter")           # Insert at specific position
print(f"After insert: {shopping}")

# Remove items
shopping.remove("bread")               # Remove first occurrence
print(f"After remove: {shopping}")

last_item = shopping.pop()             # Remove and return last item
print(f"Removed: {last_item}")
print(f"After pop: {shopping}")

# Change items
shopping[0] = "almond milk"            # Change first item
print(f"After change: {shopping}")
```

#### Tuples - Immutable Collections

```python
# Creating tuples
coordinates = (10, 20)          # Parentheses optional but recommended
rgb_color = (255, 128, 0)       # RGB color values
person = ("Alice", 25, "Engineer")  # Name, age, job

print(f"Coordinates: {coordinates}")
print(f"RGB color: {rgb_color}")
print(f"Person info: {person}")

# Accessing tuple elements (same as lists)
print(f"X coordinate: {coordinates[0]}")
print(f"Person's name: {person[0]}")
print(f"Person's age: {person[1]}")
```

#### OPTIONAL: Tuples vs Lists

```python
# Key differences between tuples and lists
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)

print("Lists vs Tuples:")
print(f"List: {my_list}, type: {type(my_list).__name__}")
print(f"Tuple: {my_tuple}, type: {type(my_tuple).__name__}")

# Lists are mutable
my_list[0] = 99
print(f"Modified list: {my_list}")

# Tuples are immutable (this would cause an error):
# my_tuple[0] = 99  # TypeError: 'tuple' object does not support item assignment

# When to use tuples:
# 1. Data that shouldn't change (coordinates, RGB values)
# 2. Multiple return values from functions
# 3. Dictionary keys (tuples can be keys, lists cannot)

print("\nTuple use cases:")
point = (3, 4)                    # Coordinates that shouldn't change
name_age = ("Bob", 30)           # Related data that belongs together
print(f"Point: {point}")
print(f"Name and age: {name_age}")
```

#### Dictionaries - Key-Value Pairs

```python
# Creating dictionaries
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8
}

# Alternative syntax
grades = dict(math=95, english=88, science=92)

print(f"Student info: {student}")
print(f"Grades: {grades}")

# Accessing dictionary values
print(f"Student name: {student['name']}")
print(f"Math grade: {grades['math']}")

# Adding new key-value pairs
student["year"] = "Junior"
print(f"Updated student: {student}")
```

#### OPTIONAL: Dictionary Methods and Operations

```python
# Dictionary methods
inventory = {"apples": 50, "bananas": 30, "oranges": 25}

print(f"Inventory: {inventory}")
print(f"Keys: {list(inventory.keys())}")        # Get all keys
print(f"Values: {list(inventory.values())}")    # Get all values
print(f"Items: {list(inventory.items())}")      # Get key-value pairs

# Safe access with get()
apple_count = inventory.get("apples", 0)        # Returns 0 if key doesn't exist
grape_count = inventory.get("grapes", 0)        # Returns 0 (default)
print(f"Apples: {apple_count}")
print(f"Grapes: {grape_count}")

# Check if key exists
print(f"Have apples: {'apples' in inventory}")
print(f"Have grapes: {'grapes' in inventory}")

# Update inventory
inventory["apples"] = 45          # Modify existing
inventory["grapes"] = 20          # Add new
del inventory["oranges"]          # Remove key-value pair
print(f"Updated inventory: {inventory}")
```

## Part 6: Putting It All Together

#### A Complete Example

```python
# Create a simple contact book
contacts = {
    "alice": {
        "phone": "555-1234",
        "email": "alice@email.com",
        "age": 28
    },
    "bob": {
        "phone": "555-5678", 
        "email": "bob@email.com",
        "age": 32
    }
}

# Display contact information
print("=== Contact Book ===")
for name, info in contacts.items():
    print(f"\nName: {name.title()}")
    print(f"  Phone: {info['phone']}")
    print(f"  Email: {info['email']}")
    print(f"  Age: {info['age']} years old")

# Add a new contact
new_contact = input("\nEnter name for new contact: ").lower()
phone = input("Enter phone number: ")
email = input("Enter email: ")
age = int(input("Enter age: "))

contacts[new_contact] = {
    "phone": phone,
    "email": email, 
    "age": age
}

print(f"\nAdded {new_contact.title()} to contacts!")
print(f"Total contacts: {len(contacts)}")
```

#### OPTIONAL: Data Validation Example

```python
# Function to validate and clean user input
def get_valid_age():
    """Get a valid age from user input"""
    while True:
        age_input = input("Enter age (18-120): ")
        
        # Check if it's a number
        if not age_input.isdigit():
            print("Please enter a valid number.")
            continue
        
        age = int(age_input)
        
        # Check if it's in valid range
        if age < 18 or age > 120:
            print("Age must be between 18 and 120.")
            continue
        
        return age

def get_valid_email():
    """Get a valid email from user input"""
    while True:
        email = input("Enter email address: ")
        
        # Basic email validation
        if "@" in email and "." in email:
            return email
        else:
            print("Please enter a valid email address.")

# Example usage (commented out to avoid input prompts)
# print("Creating a new user profile:")
# name = input("Enter your name: ")
# age = get_valid_age()
# email = get_valid_email()

# user_profile = {
#     "name": name,
#     "age": age,
#     "email": email,
#     "created": "today"
# }

# print(f"Profile created: {user_profile}")
```

#### Common Patterns and Best Practices

```python
# 1. Use meaningful variable names
student_count = 25              # Good
sc = 25                        # Bad

# 2. Use constants for values that don't change
MAX_STUDENTS = 30
MIN_AGE = 18
DEFAULT_GRADE = "A"

# 3. Group related data in dictionaries or tuples
# Good: grouped data
person = {
    "name": "John",
    "age": 25,
    "city": "New York"
}

# Less good: separate variables
person_name = "John"
person_age = 25
person_city = "New York"

# 4. Use appropriate data types
prices = [19.99, 29.99, 39.99]    # List for multiple values
coordinates = (10, 20)             # Tuple for fixed pairs
settings = {"theme": "dark"}       # Dictionary for key-value config

print("Best practices example:")
print(f"Max students allowed: {MAX_STUDENTS}")
print(f"Person info: {person}")
print(f"First price: ${prices[0]}")
```

#### OPTIONAL: Memory Usage and Performance

```python
import sys

# Compare memory usage of different data types
data_samples = [
    ("integer", 42),
    ("float", 3.14),
    ("string", "hello"),
    ("list", [1, 2, 3, 4, 5]),
    ("tuple", (1, 2, 3, 4, 5)),
    ("dict", {"a": 1, "b": 2, "c": 3}),
]

print("Memory usage comparison:")
for name, value in data_samples:
    size = sys.getsizeof(value)
    print(f"{name:8s}: {size:3d} bytes - {repr(value)}")

# Performance comparison: list vs tuple
import time

# Time list creation
start = time.time()
for _ in range(100000):
    data = [1, 2, 3, 4, 5]
list_time = time.time() - start

# Time tuple creation  
start = time.time()
for _ in range(100000):
    data = (1, 2, 3, 4, 5)
tuple_time = time.time() - start

print(f"\nPerformance comparison (100,000 creations):")
print(f"List creation:  {list_time:.4f} seconds")
print(f"Tuple creation: {tuple_time:.4f} seconds")
print(f"Tuples are {list_time/tuple_time:.1f}x faster to create")
```

---

## Summary

You've learned the fundamental building blocks of Python:

**Basic Data Types:**
- **Integers**: Whole numbers (`42`, `-5`)
- **Floats**: Decimal numbers (`3.14`, `19.99`)
- **Strings**: Text data (`"Hello"`, `'Python'`)
- **Booleans**: True/False values (`True`, `False`)

**Collections:**
- **Lists**: Ordered, changeable collections (`[1, 2, 3]`)
- **Tuples**: Ordered, unchangeable collections (`(1, 2, 3)`)
- **Dictionaries**: Key-value pairs (`{"name": "Alice"}`)

**Key Concepts:**
- Variables are labels that point to objects in memory
- Python automatically determines data types
- Use `type()` to check what type a variable is
- Convert between types using `int()`, `float()`, `str()`, `bool()`
- Choose the right data structure for your needs

**Best Practices:**
- Use descriptive variable names (`student_count` not `sc`)
- Use constants for unchanging values (`MAX_SIZE = 100`)
- Group related data in dictionaries or tuples
- Validate user input when needed

These fundamentals form the foundation for everything else you'll learn in Python!