import modal
import os
from datetime import datetime
from pathlib import PurePosixPath
from typing import Union
from image import vllm_image


test_r1_volume = modal.Volume.from_name(
    "test_r1_runs", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/runs": test_r1_volume,
}

APP_NAME = "test-r1"
app = modal.App(
    name=APP_NAME,
    volumes=VOLUME_CONFIG,
)

SINGLE_GPU_CONFIG = os.environ.get("GPU_CONFIG", "h100:1")


@app.function(
    image=vllm_image,
    gpu=SINGLE_GPU_CONFIG,
    timeout=6000,
)
def train_example(run_folder: str):
    import subprocess
    import os

    import torch
    from transformers.trainer_utils import get_last_checkpoint
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
    import re
    import random
    import logging
    from dataclasses import dataclass
    
    ########################
    # Custom dataclasses
    ########################
    @dataclass
    class ScriptArguments:
        dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
        dataset_splits: str = "train"
        tokenizer_name_or_path: str = None
    
    ########################
    # Setup logging
    ########################
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    
    ########################
    # Helper functions
    ########################
    def format_reward_func(completions, target, **kwargs):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
        
        Returns:
            list[float]: Reward scores
        """
        rewards = []

        for completion, gt in zip(completions, target):

            try:
                # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                completion = "<think>" + completion
                if random.random() < 0.1:  # 1% chance to write samples into a file
                    os.makedirs(os.path.join(run_folder, "completion_samples"), exist_ok=True)
                    log_file = os.path.join(run_folder, "completion_samples", "completion_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)
                
                # Check if the format is correct
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                match = re.search(regex, completion, re.DOTALL) 
                # if the format is not correct, reward is 0
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def equation_reward_func(completions, target, nums, **kwargs):
        """
        Evaluates completions based on:
        2. Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                completion = "<think>" + completion
                # Check if the format is correct
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
                
                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue
                
                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builtins__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                    if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                        os.makedirs(os.path.join(run_folder, "completion_samples"), exist_ok=True)
                        log_file = os.path.join(run_folder, "completion_samples", "success_completion_samples.txt")
                        with open(log_file, "a") as f:
                            f.write(f"\n\n==============\n")
                            f.write(completion)
                            
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0) 
        return rewards

    def get_checkpoint(training_args: GRPOConfig):
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        return last_checkpoint

    def grpo_function():
        model_args = ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            use_peft=True,
            load_in_4bit=True,
        )
        
        script_args = ScriptArguments()
 
        # Hyperparameters
        training_args = GRPOConfig(
            output_dir="qwen-r1-aha-moment",
            learning_rate=5e-7,
            lr_scheduler_type="cosine",
            logging_steps=10,
            max_steps=100,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            # GRPO specific parameters
            max_prompt_length=256,
            max_completion_length=1024, # max length of the generated output for our solution
            num_generations=2,
            beta=0.001,
            
        )
        
        #########################
        # Log parameters
        #########################
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Training/evaluation parameters {training_args}")

        ################
        # Load tokenizer
        ################
        tokenizer = AutoTokenizer.from_pretrained(
            (
                script_args.tokenizer_name_or_path
                if script_args.tokenizer_name_or_path
                else model_args.model_name_or_path
            ),
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ###############
        # Load datasets
        ###############
        # Load dataset from Hugging Face Hub
        dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
        # select a random subset of 50k samples
        dataset = dataset.shuffle(seed=42).select(range(50000))

        #####################
        # Prepare and format dataset
        #####################

        # gemerate r1 prompt with a prefix for the model to already start with the thinking process
        def generate_r1_prompt(numbers, target):
            r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
            },
            { 
                "role": "user",
                "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }]
            return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": numbers}

        # convert our dataset to the r1 prompt
        dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

        # split the dataset into train and test
        train_test_split = dataset.train_test_split(test_size=0.1)

        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        #########################
        # Instantiate DPO trainer
        #########################

        trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        )

        ###############
        # Training loop
        ###############
        # Check for last checkpoint
        last_checkpoint = get_checkpoint(training_args)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

        # Train the model
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
        )
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Log and save metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("*** Training complete ***")

        ##################################
        # Save model and create model card
        ##################################

        logger.info("*** Save model ***")
        trainer.model.config.use_cache = True
        trainer.save_model(os.path.join(run_folder, training_args.output_dir))
        logger.info(f"Model saved to {os.path.join(run_folder, training_args.output_dir)}")
        training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

        tokenizer.save_pretrained(os.path.join(run_folder, training_args.output_dir))
        logger.info(f"Tokenizer saved to {os.path.join(run_folder, training_args.output_dir)}")

        # Save everything else on main process
        if trainer.accelerator.is_main_process:
            trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
        # push to hub if needed
        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()

        logger.info("*** Training complete! ***")
    
    grpo_function()
    

@app.function(
    image=vllm_image,
    volumes=VOLUME_CONFIG
)
def launch():
    # Write config and data into a training subfolder.
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"test-r1-{time_string}"
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)
    
    os.makedirs(os.path.join(run_folder, "completion_samples"), exist_ok=True)
    
    print(f"Preparing training run in {run_folder}.")
    VOLUME_CONFIG["/runs"].commit()
    
    launch_handle = train_example.spawn(run_folder)

    return run_name, launch_handle


@app.local_entrypoint()
def main():
    run_name, launch_handle = launch.remote()
    # Wait for the training run to finish.
    launch_handle.get()
    print(f"Run complete.")