import os
import torch.utils.cpp_extension


if __name__ == '__main__':
    # Load file
    srcs = []
    for filename in ['build/brighter.pytorch.h', 'build/brighter_grad.pytorch.h']:
        with open(filename) as file:
            srcs.append(file.read())

    # Compile
    module = torch.utils.cpp_extension.load_inline(
        name='test_inline_ext',
        cpp_sources=srcs,
        functions=['brighter_th_', 'brighter_grad_th_'],
        extra_cflags=['-Wno-c++17-extensions', '-std=c++14', '-g'],
        extra_ldflags=[f' -L{os.getcwd()}/build ',
                       ' -lcuda ',
                       ' -l:brighter.a '
                       ' -l:brighter_grad.a '],
        with_cuda=True,
        verbose=True
    )

    # Test
    shape = [8, 8]
    device = torch.device('cuda')
    x = torch.tensor([[float(i * shape[0] + j) for j in range(shape[1])] for i in range(shape[0])], device=device)
    y = torch.empty(shape, device=device)
    module.brighter_th_(x, 8.0, y)
    print(x)
    print(y)
    y = torch.ones(shape, device=device)
    z = torch.empty(shape, device=device)
    module.brighter_grad_th_(x, 8.0, y, z)
    print(z)
