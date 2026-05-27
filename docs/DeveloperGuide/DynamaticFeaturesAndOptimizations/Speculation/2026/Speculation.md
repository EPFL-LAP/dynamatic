# Speculation

## High-Level Overview

Our token-prediction speculative flow allows the breaking of arbitrary data dependencies, and therefore reduction of performance bottlenecks. 

### Examples

#### Example 1

Take a snippet of python code as a simple example:

```python
c = f1(a)
d, e = f2(c, b)
if d:
  g = f3(e)
else:
  g = f4(e)
```

This example does not demonstrate an especially useful case of speculation, but it does demonstrate well the functionality of the individual speculative units and how they communicate.

For this example, the following fine-grained dataflow circuit would be produced:

<img alt="Pre-speculation circuit" src="./Figures/circuit_pre_spec.png" width="400" />

And an example corresponding schedule:

<img alt="Pre-speculation schedule" src="./Figures/schedule_pre_spec.png" width="600" />

With our token-prediction speculative flow, the circuit is transformed to look as follows:

<img alt="Pre-speculation circuit" src="./Figures/circuit_post_spec.png" width="400" />

Which in turn implements the following schedule instead:

<img alt="Pre-speculation circuit" src="./Figures/schedule_post_spec.png" width="600" />

In cycle 0, the speculator issues a predicted value `c'`, and the save-commit saves `b` in its internal memory and issues `b'`, allowing `f2(c', b')` to begin immediately, and perform either `f3(e')` or `f4(e')` in cycle 1. We discover if prediction was correct at the end of cycle 3. The speculator informs either `Commit 1` or `Commit 2` of this, by consuming `d'` a second time along the new red edges. In the case that the prediction of `c` was incorrect, the `Speculator` issues the correct `c`, and informs the `Save-Commit` along the green edge, so the `Save-Commit` re-issues `b`, allowing the original schedule to run in order to recover from the mis-prediction.

#### Example 2

Take the following loop circuit:

<img alt="Pre-speculation circuit" src="./Figures/loop.png" width="500" />

which has the following schedule:

<img alt="Pre-speculation circuit" src="./Figures/loop_sched.png" width="800" />

A speculator can be placed on the output of `f2` to improve loop pipelining. The output circuit is as follows:

<img alt="Pre-speculation circuit" src="./Figures/loop_spec.png" width="500" />

and produces the following schedule:

<img alt="Pre-speculation circuit" src="./Figures/loop_sched_spec.png" width="800" />

In cycle 0, the save-commit receives `i1` as the output of `f3(i0)`, and the speculator predicts the value of `f2(i0)` as `true`. This causes `i1` to speculatively re-enter the loop. 

Therefore in cycle 1, all functions execute speculatively with `i1` as their input. At the end of cycle 1, the save-commit receives a speculative `i2'` from the speculative execution of `f3(i1)`. The speculator again predicts that the output of `f2` will be `true`, causing `i2'` to re-enter the loop. Even though `i2'` is speculative, and therefore will eventually need to be resolved, the save-commit does not resolve it yet: the save-commits continue to issue values until mis-prediction is detected. This is required to allow multiple predictions to be in-flight simultaneously. 

In cycle 2, all functions execute speculatively with `i2` as their input. The real `f2(i0)` arrives at the input to the speculator, and the speculator discovers its prediction was correct. The speculator then informs the save-commits and commits of the correct prediction. Only `Commit 1` will receive this info, as only `Commit 1` will receive a value from that prediction. The speculator and save-commit allow `i3'` to speculatively re-enter the loop.

Cycle 3 is the same as cycle 2, although with the real `f2(i1)` arriving and `i4` entering the loop.

In cycle 4, the speculator discovers mis-prediction. The real value of `f2(i2)` is `false`. It informs the save-commit of the mis-prediction. The speculator issues a `false` value, and the save-commit re-issues the saved `i3`. `i3` then exits the loop. 

The speculator and save-commit both know they will receive mis-speculated input values as input: `f3(i4)` is currently arriving to the save-commit, and the speculator will still receive `f2(i3)` and `f2(i4)` in later cycles. Both units use a stateful mechanism to `kill` these values, which we describe in more detail later.

In cycles 5 and 6, the speculator sends out a `kill` communication to the commit units for the predicted `f2(i3)` and `f2(i4)`. These will both arrive at only `Commit 1`, as `Commit 2` did not receive any mis-speculated outputs for this executiont trace.


### High-Level Overview: Non-Spec vs. Spec
In our token-prediction speculative approach, there are two types of values: `non-spec` and `spec`.

If a value is `non-spec`, it means no prediction was involved in the creation of this value. Since there is no prediction, that non-existant prediction was never be resolved. This means there will never be a value that arrives from the non-existant resolution to tell us what to do. 

Whenever a `non-spec` value arrives at a point where speculation is resolved, we must automatically apply the same decisions as when we resolve a prediction to have been correct.

If a value is `spec`, a prediction was involved in its generation. When it arrives at a point where speculation is resolved, we must wait to see if it resolved correctly or incorrectly before taking any action. 

We identify `non-spec` vs `spec` data via an additional bit attached to values. 

### High-Level Overview: The Speculator

The core dataflow unit of token-prediction speculative flow is the speculator. Currently, it is placed manually, based on a user-written `#pragma dyn speculate` in the input kernel source code. The speculator can be used to break data dependencies, producing a data output before receiving a data input, by generating predicted values. These values can then be used to begin computations earlier, often resulting in improved circuit performance. The speculator has an internal history of predictions made: for a maximum of N in-flight predictions, this internal history must be at least of size N. 

When the speculator makes a prediction, it informs all the save-commit units so that the speculator and the save-commits all issue their outputs in the same cycle. The speculator may make multiple predictions before the first resolves. 

When a value arrives at the speculator to resolve speculation, the speculator informs the save-commits of the result: correct prediction or incorrect prediction. If prediction was incorrect, the speculator and save-commits re-issue their correct outputs, again all in the same cycle. 

When a prediction is discovered to have been incorrect, all in-flight predictions must also be considered mis-predicted. The speculator therefore switches mode, and sends a `kill` communication to the commit units for each in-flight prediction. The speculator does not communicate with the save-commits during this time. 

An output value of an in-flight prediction may arrive at the speculator. The speculator must therefore `kill` any incoming values which came from in-flight predictions. We describe the exact details of this later. 

### High-Level Overview: The Commit Units

The second unit used in our token-prediction speculative flow is the commit unit. Outputs of the computation done with predicted inputs will eventually reach commit units, which provide a hard boundary of how far a `spec` value can travel from the speculator. 

When the true data input arrives at the speculator, the speculator communicates to these commit units if the prediction was correct or incorrect. If correct, the commit units allows the value to `pass`. If incorrect, the commit unit `kills` the value, preventing the misprediction from impacting the correctness of the circuit's execution. We use `kill` to refer to "dropping" values which are incorrect. 

From the reasoning discussed in [Non-Spec Vs. Spec](#high-level-overview-non-spec-vs-spec), a `non-spec` value is automatically `pass`-ed by the commit unit without receiving any communication from the speculator.

### High-Level Overview: The Save-Commit Units

The third unit of the approach is the save-commit units. Save-commits are involved in speculation beginning but are not involved in speculation resolving.

The speculator may make multiple predictions before the first resolves. If so, the second value that the save-commit receives is the result of the first prediction, and so on. The save-commit 

The primary purpose of the save-commit units is to save values, to allow recovery from mis-prediction. When a computation begins with a predicted input, a set of save-commits save all other inputs to that computation (one save-commit per input). When mis-prediction is discovered, the computation must be re-executed. The speculator issues the real data input for the first time, and each save-commit re-issues their saved value. In order to have up to N in-flight predictions, the save-commits must be able to store N saved values.

The secondary purpose of the save-commit units is the "commit" purpose. This refers to how the save-commit updates its history of stored inputs after mis-prediction is detected. 

When a prediction is discovered to be correct, it means that computation will not be re-executed. The save-commit can therefore `discard` the value it was saving for that re-execution. We use `discard` to refer to 'dropping' values which were correct but are no longer needed. 

When a prediction is discovered to have been incorrect, all in-flight predictions must also be considered mis-predicted. 

An output value of an in-flight prediction may arrive at the save-commit. The save-commit must therefore `kill` any incoming values which came from in-flight predictions. The save-commit and speculator implement this in the same way, and we describe the exact details of this later. 


### High-Level Overview: How to Place Commits

Commit units are placed to limit where unresolved `spec` values can reach. In general, we place commit units to prevent `spec` values from exiting the circuit along any external connection, including being stored to memory. 

Commit units can also be placed to prevent any arbitrary computation from being performed speculatively, for any arbitrary reason.

Additional commit units are also needed for ordering correctness, but we will discuss this later.


### High-Level Overview: The Snapshot Approach (How to Place Save-Commits)

In the "snapshot approach", we use the save-commits units to store a "snapshot" of the state of the circuit at the time the prediction occured. When prediction happens, the save-commits save their incoming values to their internal history, to allow re-issuing of that snapshot. 

If mis-prediction is detected, we `kill` all computational outputs that are generated after the snapshot was created, and roll the entire circuit state back to this point, even if computation occured below the snapshot which was not affected by the mis-prediction. For multiple in-flight predictions, the snapshot approach means that any predictions made after a mis-prediction are also considered mis-predicted. 

The input values to the save-commits are often `non-spec`. However, computational outputs generated from `non-spec` inputs are `non-spec`, and `non-spec` outputs automatically pass through commit units: they cannot be `kill`-ed. The save-commits therefore issue the snapshot as `spec` when prediction happens, to guarantee that all outputs will be `spec`, and therefore all outputs will be `kill`-able if mis-prediction is detected.

In mis-prediction recovery, the speculator issues the correct value as `non-spec`, and all save-commits re-issue their values also as `non-spec`, as no prediction was involved in the issuing of these inputs. 

This guarantees that exactly one value will be generated for each output, even is a snapshot is executed twice.

If any computation receives input only from the save-commit units, and no input from the speculator, this computation will be redundantly re-executed when mis-prediction recovery happens. The placement of the save-commit must therefore trade-off the degree of redundant execution with how many save-commits must be placed to create a full snapshot.

An example below shows a single save-commit above a computation which will never be affected by prediction, with the shapshot point shown as a dashed purple line:

<img alt="Post-speculation circuit" src="./Figures/loop_with_spec_cut.png" width="600" />

The snapshot approach means the value issued by the save-commit is `spec`, so that the outputs are correctly `kill`-ed when mis-prediction is detected. `f1(ai)` then re-executes with the same inputs (but now `non-spec`) during mis-speculation recovery. 

You can see that the output of the save-commit must be marked as `spec` here. Otherwise, a `non-spec` `ai+1'` would pass through `Commit 1` twice if mis-prediction occurs when `ci+1'` is `false`. This means the circuit would produce two output values along this edge when there should only be one.

Another valid save-commit placement would be two save-commits on the two outputs of `f1(ai)`. This would reduce redundant re-computation, but would require another save-commit. 

## Individual Unit Behaviour

Here we discuss exactly how each unit functions, without extensive discussion of the properties required for this behaviour to result in correct execution.

An important fact (which we will justify more later) is that the save-commits use state to `kill` mis-speculated tokens, and so see no explicit `kill` values as input. Commit units are stateless, and so receive a `kill` or `pass` value along their `control` input for every `spec` input they receive.

### Individual Unit Behaviour: The Commit Unit

The commit unit has two inputs: `data in` and `control`, and one output: `data out`.

<img alt="Commit Unit" src="./Figures/commit_unit.png" width="400" />

The behaviour of the commit units is different for `non-spec` vs `spec values`:

<img alt="Commit Unit" src="./Figures/commit_unit_internals.png" width="400" />

A `non-spec` value has no corresponding `control`, and so passes through the commit unit immediately.

A `spec` value must join with a `control` signal to either be `pass`-ed to the output or `kill`-ed.

### Individual Unit Behaviour: The Save-Commit Unit

The save-commit unit has three inputs: `data in`, `issue control`, and `history control`, and one output: `data out`.

This diagram is the first introduction of how the speculator and save-commit statefully kill values which come from mis-speculation. If the speculator informs the save-commit of mis-prediction, the save-commit wipes its entire history and kills any incoming `spec` values until it sees a `non-spec` value. We will discuss the `non-spec` value as a flag event indicating all mis-speculated values have been killed in more detail later. 

A simplified version of how the save-commit works is shown below, with stateful aspects in purple, mis-prediction recovery aspects in red, and "normal operation" aspects in green:

<img alt="Save Commit Unit" src="./Figures/save_commit_internals.png" width="500" />

When the save-commit is not recovering from mis-prediction:
1. `data in` is stored in the history.
2. `issue control` is a value from the speculator telling the save-commit to issue a `spec` output, since the rest of the snapshot is valid.
3. `history control` is a value from the speculator telling the save-commit to `discard` its oldest saved value, as it is no longer necessary.

When mis-prediction is discovered, the speculator informs the save-commit using both `history control` and `issue control`:
1. The save-commit re-issues its oldest saved value. 
2. The entire history is wiped.
3. Incoming `spec` data is treated as mis-predicted.
4. The arrival of `non-spec` data is treated as the end of mis-prediction recovery.

This omits some details about the order in which things happen, synchronization between the two control channels, and what happens when the speculator decides not to speculate, which we will discuss in more detail later.

### Individual Unit Behaviour: The Speculator

The logic of the speculator is divided into two halves: a "brain", and a "communicator"

<img alt="Save Commit Unit" src="./Figures/speculator.png" width="800" />


The speculator "brain" unit has 5 elastic output channels: `no cmp`, `do spec`,  `resend`, `kill`, and `resolve`m as well as 3 data outputs: `in data`, `predicted data`, and `resend data`, which do not have a valid and ready signal. 

The 5 elastic output channels encode the decisions that the speculator "brain" has made. Each decision corresponds to a different set of output values that must be issued from the speculator's 4 outputs. 

The meaning of each of the five events is as follows:

- `do spec`
Perform prediction to begin speculate. Issue the predicted value and inform the save-commits so they can issue their save values.
- `no cmp`
`data in` arrived immediately and there is no need to predict. Pass through all the values as `non-spec`.
- `resolve` 
A new `data in` has arrived and matches our prediction. Tell the commit units to `pass` the outputs, and the save-commits to `discard` their oldest saved values.
- `kill`
In a previous cycle, the speculator discovered a mis-prediction. Now it sends one `kill` event per unresolved in-flight prediction.
- `resend` 
In a previous cycle, the speculator discovered a mis-prediction. `resend` is issued once to the save-commits for them to re-issue their oldest save data and wipe their history, and the speculator also issues the corrected value.

This table shows which of the decision channels can be valid simultaneously:
| `no cmp` | `do spec` | `resolve` | `kill` | `resend` |
|:-:|:-:|:---:|:---:|:---:|
| X |   |   |   |   |
|   | X |   |   |   |
|   | X | X |   |   |
|   |   | X |   |   |
|   |   |   | X |   |
|   |   |   | X | X |
|   |   |   |   | X |

`no cmp` cannot overlap with any other decision: there are no predictions to resolve. The elastic channel means the "communicator" is able to backpressure this decision. 

`do spec` and `resolve` can overlap: the speculator can make a new prediction, resolve an old prediction as correct, or both in a single cycle. The "communicator" can backpressure both decisions along the elastic channel, only one, or neither. 

`kill` and `resend` can overlap: the speculator can `kill` an in-flight mis-prediction, `resend` the correct values, or both in a single cycle. The "communicator" can backpressure both decisions along the elastic channel, only one, or neither. 

####  Individual Unit Behaviour: The Speculator's "Communicator"

The internals of the speculator's communicators look like this:

<img alt="Save Commit Unit" src="./Figures/spec_communicator.png" width="800" />

4 decoder units take different actions based on which decision the brain has taken. Each decoder is guaranteed by the mutual exclusivity of the decisions to have only a single valid input in any cycle. 

The lazy forks guarantee that if any decoder backpressures a decision, that decision is not applied. This is because the decision value from the speculator "brain" is volatile: the value can change without the handshake protocol accepting it for transfer. 

The data decoder additionally receives the 3 data values of `in data`, `predicted data` and `resend data`, and chooses which to output based on the incoming decision from the speculator `brain`. 

The Issue Control Decoder communicates with the save-commits, and sends a "issue oldest un-issued value as `spec`" for `do spec`, a "issue oldest un-issued value as `non-spec`" for `no cmp`, and "issue oldest value as `non-spec`" for `resend`.

The History Control Decoder communicates with the save-commits, and sends a "drop oldest saved value" for `no cmp`, a "drop oldest saved value" for `resolve`, and "wipe history" for `resend`.

The Commit Control Decoder communicates with the commit units, and sends a `pass` for `resolve` and a `kill` for `kill`.

####  Individual Unit Behaviour: The Speculator's "Brain"


#####  Individual Unit Behaviour: The Speculator's FSM


The core of the speculator's "brain" is a simple finite state machine.

<img alt="Save Commit Unit" src="./Figures/spec_fsm_interface.png" width="400" />

It has 3 inputs: `mis-spec detected`, `ready to re-speculate` and `non-spec data`, and 1 output: `state`.

<img alt="Save Commit Unit" src="./Figures/spec_fsm.png" width="800" />

The FSM has three states `IDLE`, `KILL` and `KILL ONLY DATA`. The speculator makes predictions in both `IDLE` and `KILL ONLY DATA`. The speculator only resolves predictions in `IDLE`. In `KILL`, the speculator sends out one `kill` value for each in-flight mis-predictions, and independently attempts to perform `resend`. In both `KILL` and `KILL ONLY DATA`, the speculator `kill`-s any incoming `spec` values, as they are the results of mis-prediction. 

The speculator transitions from `IDLE` to `KILL` when mis-prediction is detected.

An important sub signal is `ready to re-speculate`, which occurs once all `kill`-s have been sent by the "communicator", and the speculator is receiving a `non-spec` `trigger` value. This `non-spec` `trigger` value is a request from the circuit to begin a fresh round of speculation. The arrival of the `non-spec` `trigger` can only happen as a consquence of a successful `resend`, and so the `resend` event is not explicitly in the transition conditions. 

Another signal used in the other three transition is `non-spec` `data`. This `non-spec` `data in` is treated as a flag event meaning that all data values from the in-flight mis-predictions have been been killed.

The speculator transitions from `KILL` to `IDLE` when `ready to re-speculate` occurs and there is `non-spec` data: all `kill`-ing is complete.

The speculator transitions from `KILL` to `KILL ONLY DATA` when `ready to re-speculate` occurs but there is not yet `non-spec` data: some in-flight mis-predictions may not yet have been killed. As mentioned above, the speculator can start making fresh predictions in `KILL ONLY DATA`, which is helpful for performance reasons.

The speculator transitions from `KILL ONLY DATA` to `IDLE` when there is `non-spec` data: all `kill`-ing is complete.

#####  Individual Unit Behaviour: From FSM to Decision

A rough block diagram of the brain as a whole is then:

<img alt="Save Commit Unit" src="./Figures/spec_brain.png" width="600" />

The history stores unresolved in-flight predictions. 

`Ready to Respeculate` takes as input the history and the trigger. If a `kill` has been sent for each mis-prediction, the history will be empty. When the history is empty and the trigger is `non-spec`, `Ready to Respeculate` sends a value. This is used for the FSM transitions.

`Prediction Check` evaluates an incoming true value against the oldest value in the history. If mis-prediction is detected, it informs the FSM and `Resend Done`, as well as the `Output Unit`. If a the prediction was correct, only the `Output Unit` is informed.

The FSM as described before takes in three inputs and outputs the `state`. 

The `Predictor` produces a predicted value for each trigger, based on the specified prediction mechanism. 

`Resend Data Reg` stores the incoming data when mis-prediction is detected, so it can be resent properly. 

The `Output Unit` itself then encodes which decision is made based on the `state`, `trigger`, `data in`, `resend done` and the output of Prediction Check. 

A simple pseudocode is the easiest way to describe the behaviour of the Output Unit:

```
IDLE:
    if mis-prediction detected:
        store data in to resend reg
        accept trigger              # any trigger now is spec, so kill it
        set resend not done
        # FSM will move to KILL

    else:
        # if statement 1: confirm a correct speculation?
        if data matches prediction:
            emit resolve

            # backpressure from communicator?
            if resolve accepted:
                accept data in
                pop oldest prediction from history

        # if statement 2: did real data arrive before any prediction
        if data arrived before prediction:
            emit no_cmp

            # backpressure from communicator?
            if no_cmp accepted:
                accept data in
                accept trigger

        # otherwise, speculate on the new trigger
        else if trigger present and history has room:
            emit do_spec

            # backpressure from communicator?
            if do_spec accepted:
                accept trigger
                push prediction into history

KILL:
    if data in is spec:
        accept data in            # to kill it
    if trigger is spec:
        accept trigger            # to kill it

    if history not empty:
        emit kill

        # backpressure from communicator?
        if kill accepted:
            pop oldest prediction from history

    if not resend done:
        emit resend

        # backpressure from communicator?
        if resend accepted:
          set resend done

KILL_ONLY_DATA:
    if data in is spec:
        accept data in            # to kill it

    # speculate on the new trigger
    if trigger present and history has room:
        emit do_spec
        if do_spec accepted:
            accept trigger
            push prediction into history

```


## Speculator to Save-Commit Communication

The communication between the speculator and save-commit is rife with deadlock risks. 

Take for example a speculator and save-commit in a do-while loop:

<img alt="Save Commit Unit" src="./Figures/save_commit_control.png" width="600" />

Whenever backpressure is present on the data out channel of the save-commit, it cannot accept instructions from the speculator relating to issuing. However, to remove this backpressure, the save-commit may need to `discard` a value from its internal history, so it can accept a new value on the data in channel. Therefore, to avoid deadlocking, the save-commit must be able to accept a history-based instruction from the speculator even when 1) it has backpressure at its input and 2) the speculator also wants it to issue. 

In order to be able to have independent transfer of the two types of instructions, we need two handshaking channels. We call these two channels the `issue control` and the `history control`.

<img alt="Save Commit Unit" src="./Figures/issue_and_hist.png" width="600" />

When backpressure propagates to the `issue control` channel, a value can still transfer on the `history control` channel, freeing up space in the save-commit and preventing deadlock.

Two handshaking channels between two units poses a issue: there is no guaranteed relative arrival order between the two channels. However, the speculator expects its instructions to be applied in the order they are issued. Additionally, some instructions affect the internal history and also require a value to be issued. How should these instructions be communicated and applied to ensure correctness?

The solution is synchronized acceptance of problematic instructions. Any instruction which affects both the internal history and the issuing of values must be applied after all previous instructions have been succesffuly applied. To avoid the instruction overtaking a value on the other channel, we send the instruction along both channels, and only accept it once it has arrived on both channels.

Take for example this situation:

<img alt="Save Commit Unit" src="./Figures/issue_hist_order.png" width="600" />

The speculator has sent `do spec` twice along `issue control` and then `resend` on both `issue control` and `history control`. `resend` wipes the internal history and `do spec` reads from the internal history: if we apply these in the wrong order, the `do spec` instruction will have nothing to send. Synchronized acceptance means the save-commit sees the `resend` along `history control`, but does not accept it until `resend` also arrives along `issue control`.

# Cut Content

The placement of the `Save-Commit` units to form the "snapshot point" must-trade off the extent of redundant computation with the number of `Save-Commit` units required to cut every path through the circuit. 

For example, we could also form the "snapshot" point using 2 `Save-Commit` units, one on each of the outputs of `f1(ai)`, reducing the number of redundant re-executions by 1. However, since speculator can have multiple in-flight predictions, `f1` must still execute many more times than in the original schedule, as each mis-speculation resets the loop iterator back to a prior value. 

This approach simplifies consideration of how to combine speculative and non-speculative values: they are never combined, because a set of values passing through the "snapshot point" must either be all speculative or all non-speculative. 


When a prediction is discovered to be correct, the speculator informs the save-commit and commit units of this, so they can take the appropriate response. 

When mis-prediction is discovered, the effects of that prediction and also all later predictions must be `kill`-ed, and the computation must be re-executed with the correct input values. The speculator informs the save-commits of the mis-prediction once, causing the computation to receive a full set of correct inputs. The speculator then sends one `kill` communication to the commit units per unresolved prediction in its history. This causes the effects of all predictions after the mis-prediction to be `kill`-ed.

However, the speculator may still also receive mis-speculated values as input which must also be `kill`-ed. Therefore after mis-prediction is discovered, the speculator `kill`-s all incoming speculative values until a non-speculative value arrives. We discuss the arrival order of mis-speculated and non-speculative values in more detail below.

### High-Level Overview: The Commit Units


Computations performed with predicted inputs may cause different paths through the circuit to execute, and so commit units are not guaranteed to receives a value each time prediction happens. Therefore, the communication network between the speculator and the commit units must mirror the network used to deliver data inputs to the commit unit.  


### High-Level Overview: The Save-Commit Units

The secondary purpose of the save-commits is the "commit" purpose. 

When a save-commit is informed a prediction was correct, it `discards` the oldest saved value, as the computation the value belongs to will not be re-executed. We use `discard` for "dropping" values which are correct but no longer needed. 

When a save-commit is informed a prediction was incorrect, it re-issues the oldest undiscarded value, to allow the re-execution of the computation with correct inputs. The save-commit also `kills` its **entire history of saved values**, as all outputs generated after the mis-prediction are considered mis-speculated.

However, the save-commit may still also receive mis-speculated values as input which must also be `kill`-ed. Therefore after mis-prediction is discovered, the save-commit `kill`-s all incoming speculative values until a non-speculative value arrives. We discuss the arrival order of mis-speculated and non-speculative values in more detail below.

### High-Level Overview: Commit vs. Save-Commit vs. Speculator Mis-Prediction Kill Behaviour

Communication between the speculator and the save-commit for mis-prediction is simpler than communication between the speculator the commit units. 

One reason for this simpler communication is that commit units may be placed on conditional paths, and so have no guarantee they will ever execute. The sequence of `kill` or `pass` values that arrive at an arbitrary commit unit could therefore take any value, and so the exact sequence must be communicated to the commit unit.

If the speculator went multiple predictions ahead and then discovers mis-prediction, the effects of the later predictions must also be `killed`. The speculator performs a round of communication with the commit units for each prediction that must be resolved. For each round of communication, the set of commit units that executed for that specific prediction will receive the communication to `kill` the next incoming value. 

The speculator and save-commits instead execute unconditionally, that is they must all execute for every round of speculation. This provides useful information about the sequences of `pass` and `kill` that should be applied at their inputs. 

A second reason for this simpler communication is that the speculator and save-commit units do not wait for `pass` values before consuming speculative values. If they did, we would be limited to 1 in-flight speculation, as the speculator and save-commits would wait for the first speculation to resolved before beginning the next round of speculation.


If the speculator went multiple predictions ahead and then discovers mis-prediction, the speculator still only communicates with the save-commit units once. 

th the save-commit and speculator receive mis-speculated values at their input.


There have been 2 previous approaches for deciding how to save non-speculative input values in order to re-execute when mis-prediction occurs. 

The original approach saved non-speculative values using 2 different types of dataflow units, and only saved non-speculative values directly before they interacted with a speculative value. This caused complications in cases where the speculative value did not impact the control flow of the circuit. 

An example of such a circuit is below:

<img alt="Pre-speculation circuit" src="./Figures/loop_with_spec.png" width="600" />

Dynamatic does not literally use combined loop header units, however the connections between the units which make up this behaviour are not compatible with the assumptions of the original approach. 

A second approach by Haoran Zhao solved these issues by moving to 3 units for deciding how values are saved and computations re-executed. By simplifying this to the use of a single unit, we arrived at the current approach, which we call the "snapshot" approach.


This is also what allows the `Save-Commit` units to `kill` their entire input history when mis-prediction is detected. Every value generated after the prediction is considered mis-speculated, even if they were not affected by the prediction.

### High Level Overview: Re-Speculating after Mis-Prediction 

When mis-prediction occurs, the speculator and save-commits issue non-speculative values for mis-prediction recovery. 

All speculative values anywhere in the circuit are at this point considered mis-speculated, and should be killed. Fine-grained dataflow circuits do not guarantee the location or arrival time of any of these speculative values. 

However, for correct execution, we must be able to tell this set of mis-speculated values from a future round of speculation, if speculation begins again. 

Despite providing no guarantees about arrival time, with some effort we can provide guarantees about the arrival order of tokens.

Therefore, in the current implementation, we separate the two sets of values by the arrival of the non-speculative values. Values that arrived before the non-speculative data are considered mis-speculated, and and values that arrive after the non-speculative data are considered to be from a new round of speculation.

The speculator and save-commits units therefore individually exit from their mis-prediction recovery state with the arrival of a non-speculative value at their input.


## Interface

## Internal Structure



