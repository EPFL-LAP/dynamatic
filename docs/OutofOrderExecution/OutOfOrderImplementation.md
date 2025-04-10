# Out-of-Order Execution Implementation

## Step 1: Create and Connect the Operations
Out-of-Order Execution Algorithm :

1. Identify dirty nodes

2. Identify unaligned edges

3.  Identify tagged edges

4.  Add taggers

5. Add aligner + untaggers

**Example:**

Code: **[2..32] << a[0]**

Out-of-Order Node: LD

<img alt="Example diagram" src="./Figures/Example.png" width="300" />

### 1.1. Identify Dirty Nodes
Identify all the nodes reachable from the out-of-order node.

<img alt="Step 1.1 diagram" src="./Figures/Step1.1.png" width="300" />


### 1.2. Identify Unaligned Edges
Identify all the edges that can have tokens with different orders (edges between dirty and non-dirty node).

<img alt="Step 1.2 diagram" src="./Figures/Step1.2.png" width="300" />

### 1.3. Identify Tagged Edges
Identify all the edges that should receive tagged tokens.

These are the:

&nbsp;&nbsp; *unaligned edges + input edges of the out-of-order node – output edge of the out-of-order node*

<img alt="Step 1.3 diagram" src="./Figures/Step1.3.png" width="300" />

### 1.4. Add Taggers
For each tagged edge, add a tagger.

<img alt="Step 1.4 diagram" src="./Figures/Step1.4.png" width="300" />

### 1.5. Add Aligner + Untaggers
Align the unaligned edges.


Aligner Order (select):

- Free Aligner: out-of-order node order

- Controlled Aligner: program order

<img alt="Step 1.5 diagram" src="./Figures/Step1.5.png" width="600" />

## Step 2: Adding Channel Types
Traverse the tagged region to add the tag as an extra signal to the channels.

<img alt="Step 2 diagram" src="./Figures/Step2.png" width="600" />