# EigenTensor

> the easiest way to deploy work onto a remote GPU ever. oh, and the result of each computation is verifiable.

TLDR: We implemented a universal, memory-safe, cross-platform format and runtime for GPU computations, which can run efficiently on any AVS node. We achieved this by reverse-engineering the undocumented internal representation of tinygrad, a popular open-source machine learning library. This gives developers access to the full extent of each GPU's capabilities without permitting malicious code execution, offering more functionality than Kite, Autonome, and Hyperbolic. It also supports tinygrad's familiar high-level API, often used by ML engineers. The key breakthrough came from exploiting a bug in tinygrad's BUFFER UOp implementation, which lets us substitute input values into the computational graph for execution on any GPU node. We also implemented basic consensus mechanisms to ensure the result of each computation is correct.

**SPECIAL NOTE:** you can split the graphs created by this tool into parts and run them on multiple GPUs!!! it doesn't parallelize the work, but it splits the VRAM requirements across multiple GPUs, which is a big deal!! VRAM is the limiting factor when running language models most of the time. we know that this is a good idea because it's used in the popular Exo inference software!

## Prove That It Works

Check out the `anytensor/src/anytensor/cli.py` file. This is a command-line tool that allows you to compile a tinygrad program into an EigenTensor graph, and then substitute values into the graph and execute it on a remote GPU. It does this all for you, you can read the code to confirm. The fact that this works is proof that we've achieved our goals.

## Why?

Building GPU applications is hard. Whether you're deploying LLMs, doing scientific computing, or any other GPU-intensive task, you quickly run into limitations:

1. **Security Nightmares**: distributed GPU computing usually means running arbitrary code on other people's machines - this can lead to data theft, DDoS attacks, and even physical damage to the machine
2. **Leaky Abstractions**: high-level frameworks break down when you need custom functionality, leaving you to either write complex low-level code or dumb-down your application for the sake of shipping on time
3. **Cross-Platform Complexity**: getting different GPUs to behave the same way is a nightmare, especially when deterministic consensus is required   

EigenTensor solves this by taking the popular tinygrad library - known for its simple yet powerful syntax for writing memory-safe GPU code - and extending it to make it verifiable, distributed, and easy to deploy. With just ~3 lines of code added to any existing tinygrad codebase, you can:

1. Compile your GPU instructions into a universal, optimized format
2. Deploy them to any GPU node, without any risk of malicious code execution
3. Get a simple REST API for your GPU application, featuring one-click deployment

Results are automatically verified through consensus between multiple nodes. Our economic incentive model ensures nodes are rewarded for honest computation and penalized for dishonest behavior, making cheating unprofitable.

No more wrestling with GPU complexities. No more framework limitations. Just write your computation in tinygrad, and EigenTensor handles the rest.

## Repository Structure

This meta repository was made for the 2025 EigenLayer "Eigen Games" hackathon. As such, we've built a serious core library, and paired it with less polished demo code and hacky dockerfiles for easy deployment. Yes, judges, we did it within fourty-eight hours of allotted time.

- `anytensor`: the core EigenTensor library
- `anytensor-mnist-frontend`: a simple Next.js frontend for MNIST demo
- `anytensor-avs-othentic`: an EigenTensor AVS written using the Othentic SDK
- `Dockerfile`: a Dockerfile for easy demo deployment

## The TinyGrad Advantage

We discuss this in the `anytensor` README.

## How To Use EigenTensor

We discuss this in the `anytensor` README.

The crux of this API is the `TensorContext` class, which is used to define inputs to the computational graph. You only really need to know three commands:

1. `add_graph_input`: define a placeholder for an input tensor
2. `compile_to_graph`: compile the computation into a computational graph
3. `execute_graph_on_gpu`: execute the computational graph on a GPU node

```python
# Create a tensor context
context = TensorContext()

# Define placeholder tensors
input_a = context.add_graph_input("matrix_a", (1000, 1000))
input_b = context.add_graph_input("matrix_b", (1000, 1000))

# Define computation (matmul in this example)
# This can get as complex as you want, you can write entire models if you want
result = input_a @ input_b

# Export the task for later use
task = context.compile_to_graph(result)
```

Further details are available in the `anytensor` README.

## The Hackery

### Using TinyGrad To Build EigenTensor

The core innovation of EigenTensor came from discovering and exploiting a subtle implementation detail in TinyGrad's tensor representation system. 

TinyGrad operates by building computational graphs that represent operations on tensors, rather than executing operations immediately. When you add two tensors, TinyGrad creates a node in this graph connecting them with an ADD operation. The actual GPU computation only happens when you call `.realize()` on a tensor.

Our challenge was creating a system where computational graphs could be defined without specifying all input data upfront. We needed "placeholder" tensors that could be substituted with real data at execution time. This would allow us to share executable graphs that nodes could run with their own inputs.

The solution involved modifying how TinyGrad's BUFFER operation works. Normally, a BUFFER operation takes a two-element tuple that encodes tensor information like shape and size. We discovered we could extend this tuple with a third element containing a special placeholder label string, without breaking TinyGrad's internal operations. This allowed us to substitute input values into the computational graph by specifying the placeholder label.

This hack was necessary because TinyGrad's official metadata systems (like VOID patterns) were incompatible with graph composition operations - using them would have made it impossible to combine graphs involving placeholder tensors. Our approach was the only viable method we found after extensive testing.

With this technique, we could create computational graphs that referenced input tensors. These graphs could represent an entire ML model's execution process. When a node wants to run the computation, our system uses the undocumented graph-rewriting API of tinygrad to substitute the placeholders with actual tensors containing their data. We then serialize the graph using safe techniques, allowing it to be safely shared, stored, and executed later on any compatible GPU.

The consensus verification system built on top of this is inspired by my previous academic work on the EvML paper for Exo Labs, which modeled consensus for verifiable computation. This theoretical foundation informed our implementation of the economic security model that makes EigenTensor's execution trustworthy across distributed nodes.

### Building an AVS for EigenTensor with Othentic

We built an AVS for EigenTensor using the Othentic SDK. This AVS is a simple REST API that allows users to submit tasks to be executed on a remote GPU.

We leverage their inbuilt consensus system heavily.

## Economic Security Model

The security of EigenTensor relies on game-theoretic principles of EigenLayer's restaking model. 

FOR THE SAKE OF THIS HACKATHON, WE JUST USE MAJORITY VOTING AND RESULT REPLICATION. FURTHER DETAILS ON AN ALTERNATIVE APPROACH ARE GIVEN IN THE `anytensor` README.

## Limitations

1. there's no way of delegating non-tensor computations to the AVS just yet, although this can be done fairly easily. support for non-tensor computations is important for popular LLMs, since the process of tokenizing and looping over tokens until the end of the sequence is a non-tensor operation. you can still perform this client-side, it's just not efficient.
2. the consensus features are not yet formally proven, although we have tested them they appear to work as expected.
3. some aspects of the code, such as our hacking around tinygrad and the system of deployment for this codebase, are prone to future breakage in future versions of tinygrad and Tangle.

## Getting Started

```bash
# Install the package
pip install pipenv

# Install the dependencies
pipenv install

# Build the package
pipenv run tox -e build

# Enter development shell
pipenv shell

# Install the package, if it still isn't installed
pip install -e .

# Run the API server, if you have set Environment Variables
python -m anytensor.api
```
---

EigenTensor is licensed under the GPL V3 License.
