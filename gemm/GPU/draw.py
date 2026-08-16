import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent
csv_files = sorted(base.glob("sgemm_benchmark_v*.csv"))

if not csv_files:
    raise SystemExit("No sgemm_benchmark_v*.csv files found.")

plt.figure(figsize=(10, 6))

for csv_file in csv_files:
    data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
    sizes = data[:, 0]
    cublas_gflops = data[:, 1]
    mygemm_gflops = data[:, 2]

    edition = csv_file.stem.split("_")[-1]
    plt.plot(sizes, cublas_gflops, label=f"cuBLAS {edition}", marker="o")
    plt.plot(sizes, mygemm_gflops, label=f"MySGEMM {edition}", marker="s")

plt.title("CUBLAS vs MySGEMM GFLOPS (all versions)", fontsize=14)
plt.xlabel("Matrix Size (N x N)", fontsize=12)
plt.ylabel("GFLOPS", fontsize=12)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("cublas_vs_gemm_all.png", dpi=300)
plt.show()
