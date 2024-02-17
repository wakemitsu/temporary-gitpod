
from modal import Image, Stub

stub = Stub()

stub.image = Image.from_registry(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
)

# Now, we can create a function with GPU capabilities. Run this file with
# `modal run install_cuda.py`.


@stub.function(gpu="T4")
def f():
    import subprocess

    subprocess.run(["nvidia-smi"])


@stub.local_entrypoint()
def main():
    f.remote()
