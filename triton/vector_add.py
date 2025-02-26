import modal


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
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
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        
        # Triton's special syntax for launching kernels
        # The [grid] part tells Triton how to parallelize the computation
        # The parameters in () are the actual arguments passed to the kernel
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        
        return output
    
    torch.manual_seed(0)
    size = 98432
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    
    output_torch = x + y
    output_triton = add(x, y)
    
    print(output_torch)
    print(output_triton)
    
    print(f"Max difference: {torch.max(torch.abs(output_torch - output_triton))}")


@app.local_entrypoint()
def main():
    print("Starting modal function")
    modal_function.remote()
    