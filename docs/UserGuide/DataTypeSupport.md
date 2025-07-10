[Table of Contents](../README.md)

[Writing C Code for Dynamatic](WritingHLSCode.md)

# Data Type Support for Dynamatic
These types are most crucial when dealing with function parameters. Some of the unsupported types may work on local variables without any compilation errors.
> Arrays of supported data types are also supported as function parameters

| buffer algorithm/data type | Supported
|---|---|
|unsigned | ✓ |
|int32_t / int16_t / int8_t|✓|
|uint32_t / uint16_t / uint8_t|✓|
|char / unsigned char | ✓|
|short|✓|
|float|✓|
|double|✓|
|long/long long/long double | x|
|uint64_t / int64_t | x |
__int128|x|

# Operation support
|Type/Operation | int | float|int_example| float_example | working alternatives| comments|
|---|---|---|---|---|---|---|
|Incremenetation | ✓ | ✓ | `x++;`| `x++;` | `x += 1;`/`x += 0.1` |
|Decremenetation | ✓ | ✓ | `x--;`| `x--;` | `x -= 1;`/`x -= 0.1` |
|Comparison|✓ |✓| `x >= 1` | `x <= 0.45` | all comparison formats and operations are supported|
|AND|✓|x|`x && 2` | `x && 1.0f`(not supported just as in regular C)| `(x > 0.45f && x < 4.5f)` (works for compound conditions involving floats)| 
|OR and NOT| ✓ | x | `x \|\| y` | `x \|\| 0.45` (invalid) | see AND|
|Type Casting| ✓ |✓ | `(int)x` | `(float)x`| applies to all int and float/double types|
|Precompiled math functions| Only `abs`| only `fabsf`| `abs(x)` | `fabsf(x*2.0f)` | most precompiled C libraries are not supported. Consider coding custom functions for use with dynamatic as it requires requires explicit C|

> Data type and operation related errors generally state explicitly that an operation or type is not supported. Kindly report those as bugs on our repository while we work on making more data types supported.

## Other C Constructs
### Structs
`struct`s are currently not supported. Consider passing inputs individually rather than grouping with structs

### Function Inlining
The `inline` keyword is not yet supported. Consider `#define` as an alternative for inlining blocks of code into your target function

### Volatile
The `volatile` keyword is supported but has zero impact on the circuits generated. <span style="color:red;">Do not use on function parameters! </span>