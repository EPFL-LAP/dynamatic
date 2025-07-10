[Table of Contents](../README.md)

# Dressing up your C/C++ code for Dynamatic
Before passing your C kernel (function) to Dynamatic for compilation, it is important that you ensure it meets the following guidelines.

> Note that these guidelines target the function to be compiled and not the `main` function of your program except for the `CALL_KERNEL`. Main is primarily useful for passing inputs for simulation and is not compiled by Dynamatic

## Summary
1. [Dynamatic header](#1-include-the-dynamatic-integration-header)
2. [`CALL_KERNEL` macro in `main`](#2-use-the-call_kernel-macro-in-the-main-function)
3. [Inline functions called by the kernel](#3-all-functions-called-by-your-target-function-must-be-inlined)
4. [No recursive calls](#4-recursive-calls)
5. [No pointers](#5-pointers)
6. [No dynamic memory allocation](#6-dynamic-memory-allocation)
7. [Pass global variables](#7-global-variables)
8. [No support for local array declarations](#8-local-array-declarations)

## **1. Include the Dynamatic integration header**

To be able to compile in Dynamatic, your C files should include the `Integration.h` header that will be a starting point for accessing other relevant Dynamatic libraries at compile time.
```
#include "dynamatic/Integration.h"
```
## **2. Use the CALL_KERNEL macro in the main function**

The `CALL_KERNEL` macro is available through Dynamatic's integration header. 
It does two things in the compiler flow:
- Dumps the argument passed to the kernel to files in sim/INPUT_VECTORS (for C/HDL cosimulation when the `simulate` command is ran).
- Dumps the argument passed to the kernel to a profiler to determine which loops are more important to be optimized using buffer placement.
```
CALL_KERNEL(func, input_1, input_2, ... , input_n)
```

## **3. All functions called by your target function must be inlined**

The target function is the top level function to be implemented by Dynamatic. 
```
#define increment(int x) x+1; // macro for increment function

void loop(x) {
    while (x<20) {
        increment(x); // macro
    }
}  // inlined with macro definition.
```
## **4. Recursive calls**  
Like other HLS tools, Dynamatic does not support recursive function calls because:
- they are difficult to map to hardware
- have unpredictable depths and control flow
- unbounded execution
- the absence of call-stack in FPGA platforms would be too resource demanding to implement efficiently epecially without knowing the bounds ahead of time.  

## **5. Pointers**  

Pointers should not be used.  `*(x + 1) = 4;` is invalid. Use regular indexing and fixed sized arrays if need be as shown below.
```
int x[10]; // fixed sized
x[1] = 4; // non-pointer indexing
```
## **6. Dynamic memory allocation**
Dynamic memory allocation is also disallowed because it's not deterministic enough to allow enough hardware resources to be allocated at compile time.
<br/>

## **7. Global variables**
Pass global variables to functions as parameters else they will not be seen by Dynamatic at compilation and yield errors. See appropriate use below.

```
int scale = 2; 

int scaler(int scale, int number) // scale is still passed as parameter
{ 
    return number * scale;
}
```

## **8. Local Array declarations**
Local array declaration are not yet supported. Pass all arrays as parameters.

```
void convolution(unsigned char input[HEIGHT][WIDTH], unsigned char output[HEIGHT][WIDTH]) {
    
    int kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    int kernel_sum = 9;

    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += input[y + ky][x + kx] * kernel[ky + 1][kx + 1]; // one issue hear...non-affine apparently..
                                                                        // the kernel indexing is considered non-affine
                }
            }
            output[y][x] = sum / kernel_sum;
            printf("output[%d][%d] = %d\n", y, x, output[y][x]);
        }
    }
}
```
The above code will yield an error at compilation about array flattening. Pass it as a parameter to bypass the error:

```
void convolution(int kernel[3][3], unsigned char input[HEIGHT][WIDTH], unsigned char output[HEIGHT][WIDTH])
```


Dynamatic is being refined over time and is yet to support certain constructs such as local array declarations in the target function which must rather be passed as inputs. If you encounter any issue in using Dynamatic, kindly report the bug on the github repository.

In the meantime, visit our [examples](../GettingStarted/Tutorials/Introduction/Examples.md) page to see an example of using Dynamatic and our [supported data types and operations](DataTypeSupport.md) page for information on the constructs currently supported by Dynamatic.