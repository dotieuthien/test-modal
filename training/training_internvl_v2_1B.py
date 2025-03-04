import modal


sglang_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets",
        "flash_attn",
        "timm",
        "einops",
        "peft",
        "wandb",
        "deepspeed",
    )
)


@app.function(image=sglang_image)
def train():
    pass
