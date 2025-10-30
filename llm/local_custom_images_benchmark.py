"""
Custom Images Benchmark for Vision-Language Models

A standalone Python script to benchmark VLM performance using local images.
Run directly with: python local_custom_images_benchmark.py
"""

import asyncio
import aiohttp
import os
import json
import base64
import time
import numpy as np
from datetime import datetime
from pathlib import Path


# prompt_1 = '''Trích xuất hình ảnh dưới dạng JSON như sau {{"status":"", "bank_name":"", "value":"", "type":"", "category":"", "time":"", "noted":""}}
# Quy định trích xuất dữ liệu:
# - "status": Trạng thái giao dịch, chỉ trả về "success" hoặc "fail".
# - "bank_name": Tên ngân hàng hoặc nền tảng cung cấp dịch vụ. Nếu không có, trả về "".
# - "value": Số tiền giao dịch chính thức cuối cùng, sau khi giảm giá hoặc áp dụng voucher.
# - "type": Loại giao dịch:
#   - "thu" nếu số tiền được chuyển vào tài khoản.
#   - "chi" nếu số tiền được chuyển ra khỏi tài khoản hoặc số tiền âm.
# - "time": Thời gian thực hiện giao dịch. Nếu không có, trả về "".
# - "noted": Ghi chú nội dung đi kèm với giao dịch hoặc tên sản phẩm và dịch vụ được chi trả. Nếu không có, trả về "".

# Yêu cầu bắt buộc:
# - Nếu ảnh chứa nhiều giao dịch, trả về list các JSON object theo đúng thứ tự.
# - Đảm bảo mỗi giao dịch đều có đầy đủ 7 trường trên, nếu thông tin không có thì để "".
# - Định dạng JSON hợp lệ, không thêm mô tả ngoài dữ liệu JSON.'''

prompt_1 = "\nYou are a bank information extractor. And you are a Vietnamese. And you are so smart. You will check and correct the bank transaction information.\nMUST just return only json object, nothing else, no extra text, no explanation.\n\nUser raw input message: Mb stk 0918262525\nAnd the naive bank information extracted from user raw message: {\"bank_name\": \"Shinhan Bank\", \"bank_account_number\": \"0918 262525\", \"user_name\": \"\", \"phone_number\": \"0918262525\"}\n\n<requirements>\nYou will look at given a list of EXTRACTED bank information from user raw message:\n  1. MUST check all fields in the input EXTRACTED bank information, if it has wrong information in comparison with user raw message, fix it, return null if it is correct.\n  2. bank_account_number can be a pure number or a combination of numbers and letters, but MUST NOT have any space or special characters. And it often position near bank name.\n  3. phone_number is Vietnamese phone number, so it often position near some words like 'SĐT', 'Điện thoại', 'Momo', etc.\n  4. Note: Bank transaction information is often position near together, so you need to check all fields together.\n</requirements>\n\n<example>\nUser raw input message: \"ck cho e 1112006868 nguyễn tấn phát vcb nhé\"\nAnd the naive bank information extracted: {\"bank_name\": \"ACB\", \"bank_account_number\": \"1112 0068\", \"user_name\": \"Nguyễn Phát\", \"phone_number\": null}\n\nCorrected JSON:\n{\n  \"bank_name\": \"Vietcombank\",\n  \"bank_account_number\": \"1112006868\",\n  \"user_name\": \"Nguyễn Tấn Phát\",\n  \"phone_number\": null\n}\n</example>\n\n<response_schema>\n{ \n  \"bank_name\": Union[str, null],\n  \"bank_account_number\": Union[str, null],\n  \"user_name\": Union[str, null],\n  \"phone_number\": Union[str, null]\n}\n</response_schema>\n"

prompt_2 = '''Trích xuất hình ảnh và trả về kết quả dưới dạng JSON như sau:
{
  "error": {
    "errorCode": "",
    "errorMessage": ""
  },
  "data": {
    "name": "",
    "address": "",
    "licenseNumber": "",
    "chassisNumber": "",
    "engineNumber": "",
    "noSeat": "",
    "documentType": ""
  }
}

Phân loại và xác định error dựa trên hình ảnh:
- Nếu hình ảnh không phải là hình ảnh giấy đăng ký xe, mà là các hình ảnh không liên quan hoặc là hình ảnh của các loại giấy tờ khác: 
  errorCode: 10307
  errorMessage: "Image not the real vehicle registration"
- Nếu hình ảnh không đủ 4 góc của giấy đăng ký xe hoặc không chứa đầy đủ thông tin về phương tiện đăng kí như biển số, số khung, số máy:
  errorCode: 10305
  errorMessage: "Image missing corners or not fully visible" 
 
Quy định trích xuất dữ liệu:
- "name": Tên chủ xe
- "address": Địa chỉ
- "licenseNumber": Biển số đăng ký tiêu chuẩn, không bao gồm các kí tự diễn giải
- "chassisNumber": Số khung (Chassis)
- "engineNumber": Số máy (Engine)
- "brand": Nhãn hiệu (Brand)
- "noSeat": Số chỗ ngồi
- "documentType": Loại giấy tờ

Yêu cầu bắt buộc:
- Luôn luôn trả về JSON object hợp lệ, không thêm mô tả gì thêm ngoài dữ liệu JSON.
- Nêú thông tin trong ảnh không có thì các trường trong mục data là null
- Nếu hình ảnh không hợp lệ, chỉ trả về 1 trong 2 mã lỗi đã định nghĩa, không tự ý thêm mã lỗi khác.
'''

# Configuration
CONFIG = {
    "server_url": "https://dotieuthien--gpt-oss120b-vllm-openai-compatible-serve.modal.run/v1/chat/completions",
    "model_name": "openai/gpt-oss-120b",
    "images_dir": "/home/thiendo1/Desktop/test-modal/llm/images/test",
    "results_dir": "./local_benchmark_results",
    "bearer_token": "",  # Set your bearer token here, or None if not needed
}


async def make_request(session, semaphore, request_data, request_id, max_tokens=512, temperature=0.0):
    """
    Make a single streaming request and collect metrics.

    Args:
        session: aiohttp ClientSession
        semaphore: asyncio.Semaphore for concurrency control
        request_data: dict with image_name, image_url, prompt
        request_id: int request identifier
        max_tokens: int maximum tokens to generate
        temperature: float sampling temperature

    Returns:
        dict: Metrics for this request
    """
    async with semaphore:
        request_payload = {
            "model": CONFIG["model_name"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request_data["prompt"]},
                        # {"type": "image_url", "image_url": {"url": request_data["image_url"]}}
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

        metrics = {
            "request_id": request_id,
            "image_name": request_data["image_name"],
            "success": False,
            "error": None,
            "start_time": None,
            "ttft": None,  # Time to first token
            "end_time": None,
            "latency": None,  # End-to-end latency
            "output_tokens": 0,
            "generated_text": "",
            "itl": [],  # Inter-token latencies
        }

        try:
            metrics["start_time"] = time.time()

            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if CONFIG.get("bearer_token"):
                headers["Authorization"] = f"Bearer {CONFIG['bearer_token']}"

            async with session.post(
                CONFIG["server_url"],
                json=request_payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    metrics["error"] = f"HTTP {response.status}: {error_text}"
                    metrics["end_time"] = time.time()
                    if metrics["start_time"]:
                        metrics["latency"] = metrics["end_time"] - metrics["start_time"]
                    return metrics

                first_token_time = None
                last_token_time = None

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])

                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    current_time = time.time()

                                    # Record time to first token
                                    if first_token_time is None:
                                        first_token_time = current_time
                                        metrics["ttft"] = first_token_time - metrics["start_time"]
                                    else:
                                        # Record inter-token latency
                                        if last_token_time is not None:
                                            metrics["itl"].append(current_time - last_token_time)

                                    last_token_time = current_time
                                    metrics["output_tokens"] += 1
                                    metrics["generated_text"] += content

                        except json.JSONDecodeError:
                            continue

            metrics["end_time"] = time.time()
            metrics["latency"] = metrics["end_time"] - metrics["start_time"]
            metrics["success"] = True

        except Exception as e:
            metrics["error"] = str(e)
            metrics["end_time"] = time.time()
            if metrics["start_time"]:
                metrics["latency"] = metrics["end_time"] - metrics["start_time"]

        return metrics


def load_images(images_dir, num_prompts=None):
    """
    Load and encode images from directory.

    Args:
        images_dir: str path to images directory
        num_prompts: int optional limit on number of images

    Returns:
        list: List of dicts with image_name, image_url, prompt
    """
    print(f"Loading images from: {images_dir}")

    # List all image files
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Found {len(image_files)} images")

    # Limit if specified
    if num_prompts is not None:
        image_files = image_files[:num_prompts]
        print(f"Using {len(image_files)} images")

    # Encode images to base64
    print("Encoding images to base64...")
    requests_data = []

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)

        # Read and encode image
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine image type
        if img_file.lower().endswith('.png'):
            img_type = "png"
        else:
            img_type = "jpeg"

        image_url = f"data:image/{img_type};base64,{img_data}"

        requests_data.append({
            "image_name": img_file,
            "image_url": image_url,
            "prompt": prompt_1
        })

    print(f"Prepared {len(requests_data)} requests\n")
    return requests_data


async def run_benchmark(
    requests_data,
    max_concurrency=10,
    max_tokens=512,
    temperature=0.0,
):
    """
    Run the benchmark with concurrent requests.

    Args:
        requests_data: list of dicts with image data
        max_concurrency: int max concurrent requests
        max_tokens: int max tokens per request
        temperature: float sampling temperature

    Returns:
        tuple: (results list, benchmark_duration)
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        ttl_dns_cache=300,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"Starting benchmark with {max_concurrency} concurrent requests...")
        benchmark_start_time = time.time()

        # Create all tasks
        tasks = [
            make_request(session, semaphore, req_data, i, max_tokens, temperature)
            for i, req_data in enumerate(requests_data)
        ]

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)

        benchmark_duration = time.time() - benchmark_start_time

    return results, benchmark_duration


def print_metrics(results, benchmark_duration, max_concurrency):
    """
    Calculate and print benchmark metrics.

    Args:
        results: list of result dicts from requests
        benchmark_duration: float total benchmark time
        max_concurrency: int max concurrent requests
    """
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print(f" Custom Images Benchmark Results ")
    print(f"{'='*60}")
    print(f"{'Successful requests:':<40} {len(successful):<10}")
    print(f"{'Failed requests:':<40} {len(failed):<10}")
    print(f"{'Maximum request concurrency:':<40} {max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40} {benchmark_duration:<10.2f}")

    if successful:
        # Calculate TTFT metrics
        ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
        if ttfts:
            print(f"\n{'-'*60}")
            print(f" Time to First Token (TTFT)")
            print(f"{'-'*60}")
            print(f"{'Mean TTFT (ms):':<40} {np.mean(ttfts) * 1000:<10.2f}")
            print(f"{'Median TTFT (ms):':<40} {np.median(ttfts) * 1000:<10.2f}")
            print(f"{'P99 TTFT (ms):':<40} {np.percentile(ttfts, 99) * 1000:<10.2f}")

        # Calculate ITL metrics
        all_itls = []
        for r in successful:
            all_itls.extend(r["itl"])

        if all_itls:
            print(f"\n{'-'*60}")
            print(f" Inter-Token Latency (ITL)")
            print(f"{'-'*60}")
            print(f"{'Mean ITL (ms):':<40} {np.mean(all_itls) * 1000:<10.2f}")
            print(f"{'Median ITL (ms):':<40} {np.median(all_itls) * 1000:<10.2f}")
            print(f"{'P99 ITL (ms):':<40} {np.percentile(all_itls, 99) * 1000:<10.2f}")

        # Calculate E2EL metrics
        latencies = [r["latency"] for r in successful]
        print(f"\n{'-'*60}")
        print(f" End-to-End Latency (E2EL)")
        print(f"{'-'*60}")
        print(f"{'Mean E2EL (ms):':<40} {np.mean(latencies) * 1000:<10.2f}")
        print(f"{'Median E2EL (ms):':<40} {np.median(latencies) * 1000:<10.2f}")
        print(f"{'P99 E2EL (ms):':<40} {np.percentile(latencies, 99) * 1000:<10.2f}")

        # Calculate TPOT metrics
        tpots = []
        for r in successful:
            if r["output_tokens"] > 1 and r["ttft"] is not None:
                tpot = (r["latency"] - r["ttft"]) / (r["output_tokens"] - 1)
                tpots.append(tpot)

        if tpots:
            print(f"\n{'-'*60}")
            print(f" Time per Output Token (TPOT)")
            print(f"{'-'*60}")
            print(f"{'Mean TPOT (ms):':<40} {np.mean(tpots) * 1000:<10.2f}")
            print(f"{'Median TPOT (ms):':<40} {np.median(tpots) * 1000:<10.2f}")
            print(f"{'P99 TPOT (ms):':<40} {np.percentile(tpots, 99) * 1000:<10.2f}")

        # Throughput metrics
        total_output_tokens = sum(r["output_tokens"] for r in successful)
        print(f"\n{'-'*60}")
        print(f" Throughput Metrics")
        print(f"{'-'*60}")
        print(f"{'Request throughput (req/s):':<40} {len(successful) / benchmark_duration:<10.2f}")
        print(f"{'Output token throughput (tok/s):':<40} {total_output_tokens / benchmark_duration:<10.2f}")
        print(f"{'Total output tokens:':<40} {total_output_tokens:<10}")

    # Print failed requests
    if failed:
        print(f"\n{'-'*60}")
        print(f" Failed Requests")
        print(f"{'-'*60}")
        for r in failed:
            print(f"  [{r['request_id']}] {r['image_name']}: {r['error']}")

    print(f"{'='*60}\n")


def save_results(results, benchmark_duration, max_concurrency, max_tokens, temperature, timestamp):
    """
    Save benchmark results to JSON file.

    Args:
        results: list of result dicts
        benchmark_duration: float total benchmark time
        max_concurrency: int max concurrent requests
        max_tokens: int max tokens per request
        temperature: float sampling temperature
        timestamp: str timestamp for this run
    """
    successful = [r for r in results if r["success"]]

    # Create results directory
    results_dir = Path(CONFIG["results_dir"]) / timestamp / "custom_images"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / "benchmark_results.json"

    # Calculate summary statistics
    summary = {
        "timestamp": timestamp,
        "benchmark_duration": benchmark_duration,
        "server_url": CONFIG["server_url"],
        "model": CONFIG["model_name"],
        "num_images": len(results),
        "max_concurrency": max_concurrency,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
    }

    if successful:
        ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
        all_itls = []
        for r in successful:
            all_itls.extend(r["itl"])
        latencies = [r["latency"] for r in successful]
        tpots = []
        for r in successful:
            if r["output_tokens"] > 1 and r["ttft"] is not None:
                tpot = (r["latency"] - r["ttft"]) / (r["output_tokens"] - 1)
                tpots.append(tpot)

        total_output_tokens = sum(r["output_tokens"] for r in successful)

        summary.update({
            "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else None,
            "median_ttft_ms": float(np.median(ttfts)) * 1000 if ttfts else None,
            "p99_ttft_ms": float(np.percentile(ttfts, 99)) * 1000 if ttfts else None,
            "mean_itl_ms": float(np.mean(all_itls)) * 1000 if all_itls else None,
            "median_itl_ms": float(np.median(all_itls)) * 1000 if all_itls else None,
            "p99_itl_ms": float(np.percentile(all_itls, 99)) * 1000 if all_itls else None,
            "mean_e2el_ms": float(np.mean(latencies)) * 1000,
            "median_e2el_ms": float(np.median(latencies)) * 1000,
            "p99_e2el_ms": float(np.percentile(latencies, 99)) * 1000,
            "mean_tpot_ms": float(np.mean(tpots)) * 1000 if tpots else None,
            "median_tpot_ms": float(np.median(tpots)) * 1000 if tpots else None,
            "p99_tpot_ms": float(np.percentile(tpots, 99)) * 1000 if tpots else None,
            "request_throughput": len(successful) / benchmark_duration,
            "output_token_throughput": total_output_tokens / benchmark_duration,
            "total_output_tokens": total_output_tokens,
        })

    result_data = {
        "summary": summary,
        "detailed_results": results
    }

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"Results saved to: {result_file}")
    return result_file


async def main_async(
    num_prompts=None,
    max_concurrency=10,
    max_tokens=512,
    temperature=0.0,
    save_to_file=True,
):
    """
    Main async function to run the benchmark.

    Args:
        num_prompts: int optional limit on number of images
        max_concurrency: int max concurrent requests
        max_tokens: int max tokens per request
        temperature: float sampling temperature
        save_to_file: bool whether to save results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("Custom Images Benchmark for Vision-Language Models")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"Server: {CONFIG['server_url']}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Images directory: {CONFIG['images_dir']}")
    print(f"Max concurrency: {max_concurrency}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print("="*60 + "\n")

    # Load images
    requests_data = load_images(CONFIG["images_dir"], num_prompts)

    # Run benchmark
    results, benchmark_duration = await run_benchmark(
        requests_data,
        max_concurrency=max_concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Print metrics
    print_metrics(results, benchmark_duration, max_concurrency)

    # Save results
    if save_to_file:
        result_file = save_results(
            results,
            benchmark_duration,
            max_concurrency,
            max_tokens,
            temperature,
            timestamp,
        )

    print("="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)


def main():
    """
    Main entry point.
    """
    # You can customize these parameters
    asyncio.run(main_async(
        num_prompts=None,        # None = use all images
        max_concurrency=10,      # Number of concurrent requests
        max_tokens=512,          # Max tokens to generate
        temperature=0.0,         # 0.0 = greedy sampling
        save_to_file=True,       # Save results to JSON
    ))


if __name__ == "__main__":
    main()