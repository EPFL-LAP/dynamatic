[Table of Contents](../README.md)
# Optimizations  
Based on your needs and applications, you may consider using the following options available in Dynamtic to optimize the generated dataflow circuits to meet your goals.

## 1. Timing
Dynamatic offers compile flags with values to optimize timing

### `--buffer-algorithm`
There are two buffer placement algorithms available for this flag, `fpga20` and `fpl22`. The `fpl22` algorithm is **throughput-** and **timing-**driven whereas `fpga20` is only throughput-driven. Use it to improve the timing and throughput of your circuit.

## 2. Area  
Circuit area can be optimized using the following compile flags
- LSQ sizing
- Credit-based resource sharing: `--sharing`
- Buffer placement :`--buffer-algorithm` with value `fpl22` 

## 3. Latency and throughput
Latency and throughput can be improved using buffer placement with either the `fpga20` or `fpl22` values for the `--buffer-algorithm` compile flag.  

For more details on the optimization tools, see the [optimization tools](../DeveloperGuide/OptimizationTools.md) page.