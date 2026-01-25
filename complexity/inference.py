import torch
import os
import time
from datetime import datetime
import numpy as np
import gc
import tracemalloc


def clear_and_wait(wait_seconds=3):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    time.sleep(wait_seconds)


def measure_latency(model, img_h, img_w, batch_size, num_iterations=100, device='cuda'):
    input1 = torch.randn(batch_size, 3, img_h, img_w).to(device)
    input2 = torch.randn(batch_size, 3, img_h, img_w).to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(10):
            _ = model(input1, input2)

    latencies = []

    with torch.no_grad():
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(input1, input2)

            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            latency = (end_time - start_time) * 1000
            latencies.append(latency)

    avg_latency = np.mean(latencies)

    del input1, input2
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return latencies, avg_latency


def measure_memory(model, img_h, img_w, batch_size, device='cuda'):
    gc.collect()

    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        input1 = torch.randn(batch_size, 3, img_h, img_w).to(device)
        input2 = torch.randn(batch_size, 3, img_h, img_w).to(device)

        model.eval()
        with torch.no_grad():
            _ = model(input1, input2)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        del input1, input2
        torch.cuda.empty_cache()


    else:

        import psutil
        process = psutil.Process(os.getpid())
        gc.collect()
        mem_samples = []
        for _ in range(50):
            mem_samples.append(process.memory_info().rss / 1024 / 1024)

            time.sleep(0.001)

        input1 = torch.randn(batch_size, 3, img_h, img_w)
        input2 = torch.randn(batch_size, 3, img_h, img_w)

        model.eval()
        with torch.no_grad():
            mem_samples.append(process.memory_info().rss / 1024 / 1024)

            _ = model(input1, input2)

            mem_samples.append(process.memory_info().rss / 1024 / 1024)

        peak_memory = max(mem_samples)

        del input1, input2

        gc.collect()

    return peak_memory


if __name__ == '__main__':
    from model import embed_net_simi as embed_net

    # ==================== 参数配置 ====================
    n_class = 200
    arch = 'resnet50'
    lnr = 0.5
    cr = 32.0
    pr = 2.0
    attpos = [0, 4, 6, 0]
    img_h = 256
    img_w = 256

    checkpoint_path = '/best/DN348/epoch_best.t'
    # checkpoint_path = '/baseline/DN348/epoch_best.t'
    num_iterations = 100
    log_dir = '/DN348_best'
    wait_seconds = 3


    batch_sizes = [16, 32, 64, 128]
    os.makedirs(log_dir, exist_ok=True)

    model = embed_net(n_class, arch=arch, lnr=lnr, cr=cr, pr=pr, attpos=attpos, imgh=img_h, imgw=img_w)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['net'], strict=False)

    results = {
        'batch_size': [],
        'gpu_latency_avg': [],
        'cpu_latency_avg': [],
        'gpu_memory_peak': [],
        'cpu_memory_peak': []
    }

    for bs in batch_sizes:
        print(f"\n{'=' * 60}")
        print(f"Testing Batch Size: {bs}")
        print('=' * 60)

        clear_and_wait(wait_seconds)

        try:
            model.cuda()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            gpu_latencies, gpu_lat_avg = measure_latency(model, img_h, img_w, bs, num_iterations, device='cuda')

            clear_and_wait(wait_seconds=1)

            gpu_mem_peak = measure_memory(model, img_h, img_w, bs, device='cuda')

        except RuntimeError as e:
            gpu_lat_avg = float('nan')
            gpu_mem_peak = float('nan')
            gpu_latencies = []

        model.cpu()
        clear_and_wait(wait_seconds)
        cpu_latencies, cpu_lat_avg = measure_latency(model, img_h, img_w, bs, num_iterations, device='cpu')

        gc.collect()
        time.sleep(1)
        cpu_mem_peak = measure_memory(model, img_h, img_w, bs, device='cpu')

        results['batch_size'].append(bs)
        results['gpu_latency_avg'].append(gpu_lat_avg)
        results['cpu_latency_avg'].append(cpu_lat_avg)
        results['gpu_memory_peak'].append(gpu_mem_peak)
        results['cpu_memory_peak'].append(cpu_mem_peak)

        if gpu_latencies:
            gpu_txt = os.path.join(log_dir, f'gpu_latency_bs{bs}.txt')
            with open(gpu_txt, 'w') as f:
                for t in gpu_latencies:
                    f.write(f"{t:.6f}\n")

        cpu_txt = os.path.join(log_dir, f'cpu_latency_bs{bs}.txt')
        with open(cpu_txt, 'w') as f:
            for t in cpu_latencies:
                f.write(f"{t:.6f}\n")


    # ==================== 汇总表格 ====================
    print("\n" + "=" * 95)
    print("Summary Table (Latency: 100-iter avg, Memory: single inference peak)")
    print("=" * 95)
    print(
        f"{'Batch Size':>12} | {'GPU Latency (ms)':>18} | {'CPU Latency (ms)':>18} | {'GPU Mem (MB)':>14} | {'CPU Mem (MB)':>14}")
    print("-" * 95)
    for i, bs in enumerate(results['batch_size']):
        print(
            f"{bs:>12} | {results['gpu_latency_avg'][i]:>18.3f} | {results['cpu_latency_avg'][i]:>18.3f} | {results['gpu_memory_peak'][i]:>14.2f} | {results['cpu_memory_peak'][i]:>14.2f}")

    summary_file = os.path.join(log_dir, 'summary.csv')
    with open(summary_file, 'w') as f:
        f.write("batch_size,gpu_latency_avg_ms,cpu_latency_avg_ms,gpu_memory_peak_mb,cpu_memory_peak_mb\n")
        for i, bs in enumerate(results['batch_size']):
            f.write(
                f"{bs},{results['gpu_latency_avg'][i]:.3f},{results['cpu_latency_avg'][i]:.3f},{results['gpu_memory_peak'][i]:.2f},{results['cpu_memory_peak'][i]:.2f}\n")

    # ==================== 保存日志 ====================
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file = os.path.join(log_dir, 'measurement.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Measurement Log\n")
        f.write(f"================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Input Size: {img_h} x {img_w}\n")
        f.write(f"Iterations: {num_iterations}\n")
        f.write(f"Batch Sizes: {batch_sizes}\n")
        f.write(f"Wait Time Between Tests: {wait_seconds} seconds\n\n")
        f.write(f"Latency: 100-iteration average\n")
        f.write(f"Memory (GPU): single inference peak (torch.cuda.max_memory_allocated)\n")
        f.write(f"Memory (CPU): single inference peak (tracemalloc)\n")

    print(f"日志已保存: {log_file}")
    print("所有测试完成!")