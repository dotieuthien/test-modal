import modal


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "torch",
        "triton"
    )
)


app = modal.App("vector-add")


@app.function(
    image=image,
    gpu="L4:1"
)
def modal_function():
    global triton, tl
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each block processes BLOCK_SIZE elements:
        # If pid=0, processes elements 0-1023
        # If pid=1, processes elements 1024-2047
        pid = tl.program_id(0)

        # Calculates the starting index for this block. For example:
        # Block 0 starts at index 0
        # Block 1 starts at index 1024
        block_start = pid * BLOCK_SIZE

        # Create a range of indices [0, 1, 2, ..., BLOCK_SIZE-1]
        # Block 0 handles indices [0, 1, 2, ..., 1023]
        # Block 1 handles indices [1024, 1025, 1026, ..., 2047]
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Creates a boolean mask to handle the case
        # where the total number of elements isn't perfectly divisible by BLOCK_SIZE
        mask = offsets < n_elements

        # Loads the elements from x and y at the specified offsets
        # If the offset is out of bounds, the mask will be False
        # and the load will be ignored
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        # Performs the element-wise addition
        output = x + y

        # Stores the result back in the output tensor
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        n_elements = x.numel()

        # This lambda function defines how many parallel blocks of threads we want to launch
        # We need ceil(98432/1024) = 97 blocks to process all elements
        def grid(meta): return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        # Triton's special syntax for launching kernels
        # The [grid] part tells Triton how to parallelize the computation
        # The parameters in () are the actual arguments passed to the kernel
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

        return output

    torch.manual_seed(0)
    size = 98432
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")

    # Warmup
    for _ in range(10):
        _ = x + y
        _ = add(x, y)

    # Measure PyTorch performance
    torch.cuda.synchronize()
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)
    start_torch.record()
    for _ in range(100):
        output_torch = x + y
    end_torch.record()
    torch.cuda.synchronize()
    torch_time = start_torch.elapsed_time(end_torch) / 100

    # Measure Triton performance
    torch.cuda.synchronize()
    start_triton = torch.cuda.Event(enable_timing=True)
    end_triton = torch.cuda.Event(enable_timing=True)
    start_triton.record()
    for _ in range(100):
        output_triton = add(x, y)
    end_triton.record()
    torch.cuda.synchronize()
    triton_time = start_triton.elapsed_time(end_triton) / 100

    print(f"\nLatency comparison:")
    print(f"PyTorch: {torch_time:.3f} ms")
    print(f"Triton:  {triton_time:.3f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    print(
        f"\nMax difference: {torch.max(torch.abs(output_torch - output_triton))}")


@app.local_entrypoint()
def main():
    print("Starting modal function")
    modal_function.remote()
