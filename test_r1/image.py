import modal


cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install(
        "torch==2.5.1", 
        "transformers==4.48.1", 
        "datasets==3.1.0", 
        "accelerate==1.3.0", 
        "deepspeed==0.15.4", 
        "trl==0.14.0",
        "vllm==0.7.0",
        "peft==0.14.0",
        "bitsandbytes==0.45.0"
    )
    .entrypoint([])
)