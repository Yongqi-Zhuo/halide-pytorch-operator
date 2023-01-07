# Create a PyTorch operator using Halide without Generator

This example shows how to compile a Halide pipeline to a static library without using a Halide Generator and how to load it in PyTorch.

## Prerequisites

Install Halide and PyTorch, as well as CUDA toolkit.

Then comes a bug of Halide. In `HalidePyTorchHelpers.h` (this file is in `/usr/include/` in my system, and it may vary), there is a forward declaration of `halide_cuda_device_interface`:
```C++
// Forward declare the cuda_device_interface, for tensor wrapper.
const halide_device_interface_t *halide_cuda_device_interface();
```

But if you follow the next steps, you may find the linker complaining `undefined symbol _Z28halide_cuda_device_interfacev`. But actually if you inspect the compiled static library, you can find that `halide_cuda_device_interface` is in the symbol table, which means it is in C ABI and is mistakenly mangled by C++. So the solution is to add `extern "C"` before the forward declaration:
```C++
// Forward declare the cuda_device_interface, for tensor wrapper.
extern "C" const halide_device_interface_t *halide_cuda_device_interface();
```

So you may need to manually modify the system headers.

## Compile and Test

First build the binary and run it.
```bash
mkdir build && cd build
cmake ..
make
./gen
```

Then return to the project root directory and test the PyTorch operator.
```bash
python test.py
```

## Acknowledgement

This example is based on [this](https://github.com/LyricZhao/HyTorch) and the Halide official [tutorials](https://github.com/halide/Halide/tree/main/apps/HelloPyTorch).

## License

This example is free for use.
