## Installation

### Requirements

- Linux or Apple Silicon macOS 14+ with Python ≥ 3.10, PyTorch ≥ 2.14.1 and matching [TorchVision](https://github.com/pytorch/vision/) ≥ 0.29.1. Install the pair together following https://pytorch.org.
- For the optional NVIDIA CUDA extension: [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your installed PyTorch wheel. Check `torch.version.cuda`; the demo container uses CUDA 12.6. Rebuild the extension after upgrading PyTorch.
- If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

Then, install SAM 2 from the root of this repository via
```bash
pip install -e ".[notebooks]"
```

Note that you may skip building the SAM 2 CUDA extension during installation via environment variable `SAM2_BUILD_CUDA=0`, as follows:
```bash
# skip the SAM 2 CUDA extension
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```
On macOS this is already the default. Explicit `SAM2_BUILD_CUDA=1` requests
a CUDA build on any platform. Disabling CUDA also avoids importing its build
tooling, including when `SAM2_BUILD_ALLOW_ERRORS=0`.

Small-hole and speckle cleanup remains available: connected components use the
CUDA extension when available and suitable, otherwise SciPy on CPU. Only this
post-processing stage transfers masks to CPU; its labels and area maps return
to the original device. SciPy is installed as a core dependency.

### Apple Silicon macOS

Use native arm64 Python and an MPS-enabled PyTorch installation on macOS 14 or
later. Run locally: Docker on macOS does not expose MPS. Create a project virtual
environment, then install without any CUDA variables or toolkit:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e .
python -c 'import torch; print(torch.backends.mps.is_available())'
```

The last command must print `True` for GPU inference. Builders select MPS
automatically when CUDA is unavailable. Pass `device="mps"` to require it or
`device="cpu"` to choose CPU. Both strings and `torch.device` objects are accepted.
MPS inference and stored video memory use float32. Use
`sam2.utils.device.inference_context(model.device)` with `torch.inference_mode()`
for backend-appropriate precision; CUDA-only autocast is unnecessary on MPS.
`vos_optimized=True` and `compile_image_encoder=True` require CUDA and raise
before model construction on other backends.

Video frames default to CPU storage on MPS to reduce GPU memory use. Override
with `init_state(..., offload_video_to_cpu=False)` to store them on the selected
device. `offload_state_to_cpu=True` also stores tracking state on CPU. Both
synchronous and asynchronous JPEG loaders normalize in float32 before transfer.
JPEG directories need no video decoder; MP4 input requires the optional
`av>=13.0.0` dependency (`pip install 'av>=13.0.0'`), included in the
notebook and interactive-demo extras. PyAV provides native Apple Silicon wheels.

Start with the tiny checkpoint and a short clip. MPS results can differ
numerically from CPU/CUDA, and memory use depends on model, frames, and objects.
The required PyTorch baseline supports native MPS bicubic interpolation and
the strided pooling used by Hiera. No positional-grid CPU transfer or pooling
copy is needed. No blanket `PYTORCH_ENABLE_MPS_FALLBACK=1` setting is needed
for the validated workload.
Training remains governed by the separate CUDA/CPU trainer.

To reproduce image/automatic-mask/two-object-video qualification against CPU,
download the official tiny checkpoint and run from the repository root:

```bash
curl -L --fail -o checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
python -m tools.validate_inference --device mps --reference-device cpu \
  --output /tmp/sam2-qualification.json
```

The command checks fixed mask-agreement thresholds, reset/reuse, three sessions,
and retained GPU allocations, and reports synchronized timings, input settings,
environment and checkpoint hash. It exits nonzero on a failed gate. This is a
small regression workload, not an accuracy benchmark, peak-memory measurement,
or guarantee for larger checkpoints. See [qualification evidence](docs/apple-silicon.md).

### Building the SAM 2 CUDA extension

By default, we allow the installation to proceed even if the SAM 2 CUDA extension fails to build. (In this case, the build errors are hidden unless using `-v` for verbose output in `pip install`.)

A `Failed to build the SAM 2 CUDA extension` installation warning leaves image
and video inference available with CPU connected components. A runtime
`Skipping the post-processing step` warning indicates a separate cleanup failure;
inspect its preceding exception rather than treating cleanup as successful.

If you would like to accelerate this post-processing step, you can reinstall SAM 2 on a CUDA GPU machine with environment variable `SAM2_BUILD_ALLOW_ERRORS=0` to force building the CUDA extension (and raise errors if it fails to build), as follows
```bash
pip uninstall -y SAM-2 && \
rm -f ./sam2/*.so && \
SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e ".[notebooks]"
```

Note that PyTorch needs to be installed first before building the SAM 2 CUDA extension. Install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match `torch.version.cuda`, then check the toolkit with `nvcc --version`. Build against the same PyTorch installation used at runtime with `pip install --no-build-isolation -e .`.

Please check the section below on common installation issues if the CUDA extension fails to build during installation or load at runtime.

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
I got `ImportError: cannot import name '_C' from 'sam2'`
</summary>
<br/>

The extension is optional. Current image/video cleanup automatically uses SciPy
when it cannot be imported. Direct imports of `sam2._C` still require a successful
CUDA build; they are not part of the portable inference path.
</details>

<details>
<summary>
I got `MissingConfigException: Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'`
</summary>
<br/>

This is usually because you haven't run the `pip install -e .` step above, so `sam2` isn't in your Python's `sys.path`. Please run this installation step. In case it still fails after the installation step, you may try manually adding the root of this repo to `PYTHONPATH` via
```bash
export SAM2_REPO_ROOT=/path/to/sam2  # path to this repo
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"
```
to manually add `sam2_configs` into your Python's `sys.path`.

</details>

<details>
<summary>
I got `RuntimeError: Error(s) in loading state_dict for SAM2Base` when loading the new SAM 2.1 checkpoints
</summary>
<br/>

This is likely because you have installed a previous version of this repo, which doesn't have the new modules to support the SAM 2.1 checkpoints yet. Please try the following steps:

1. pull the latest code from the `main` branch of this repo
2. run `pip uninstall -y SAM-2` to uninstall any previous installations
3. then install the latest repo again using `pip install -e ".[notebooks]"`

In case the steps above still don't resolve the error, please try running in your Python environment the following
```python
from sam2.modeling import sam2_base

print(sam2_base.__file__)
```
and check whether the content in the printed local path of `sam2/modeling/sam2_base.py` matches the latest one in https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py (e.g. whether your local file has `no_obj_embed_spatial`) to indentify if you're still using a previous installation.

</details>

<details>
<summary>
My installation failed with `CUDA_HOME environment variable is not set`
</summary>
<br/>

This usually happens because the installation step cannot find the CUDA toolkits (that contain the NVCC compiler) to build a custom CUDA kernel in SAM 2. Please install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) or the version that matches the CUDA version for your PyTorch installation. If the error persists after installing CUDA toolkits, you may explicitly specify `CUDA_HOME` via
```
export CUDA_HOME=/usr/local/cuda  # change to your CUDA toolkit path
```
and rerun the installation.

Also, you should make sure
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
print `(True, a directory with cuda)` to verify that the CUDA toolkits are correctly set up.

If you are still having problems after verifying that the CUDA toolkit is installed and the `CUDA_HOME` environment variable is set properly, you may have to add the `--no-build-isolation` flag to the pip command:
```
pip install --no-build-isolation -e .
```

</details>

<details>
<summary>
I got `undefined symbol: _ZN3c1015SmallVectorBaseIjE8grow_podEPKvmm` (or similar errors)
</summary>
<br/>

This usually happens because you have multiple versions of dependencies (PyTorch or CUDA) in your environment. During installation, the SAM 2 library is compiled against one version library while at run time it links against another version. This might be due to that you have different versions of PyTorch or CUDA installed separately via `pip` or `conda`. You may delete one of the duplicates to only keep a single PyTorch and CUDA version.

Install the required PyTorch/TorchVision pair in one project environment, then rebuild the extension against that environment using `--no-build-isolation`. Remove an obsolete extension only after identifying its location. Downgrading below the required PyTorch baseline is unsupported; use a toolkit and wheel combination published by PyTorch instead.
</details>

<details>
<summary>
I got `CUDA error: no kernel image is available for execution on the device`
</summary>
<br/>

A possible cause could be that the CUDA kernel is somehow not compiled towards your GPU's CUDA [capability](https://developer.nvidia.com/cuda-gpus). This could happen if the installation is done in an environment different from the runtime (e.g. in a slurm system).

You can try pulling the latest code from the SAM 2 repo and running the following
```
export TORCH_CUDA_ARCH_LIST=9.0 8.0 8.6 8.9 7.0 7.2 7.5 6.0`
```
to manually specify the CUDA capability in the compilation target that matches your GPU.
</details>

<details>
<summary>
I got `RuntimeError: No available kernel. Aborting execution.` (or similar errors)
</summary>
<br/>

This is probably because your machine doesn't have a GPU or a compatible PyTorch version for Flash Attention (see also https://discuss.pytorch.org/t/using-f-scaled-dot-product-attention-gives-the-error-runtimeerror-no-available-kernel-aborting-execution/180900 for a discussion in PyTorch forum). You may be able to resolve this error by replacing the line
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
```
in [`sam2/modeling/sam/transformer.py`](sam2/modeling/sam/transformer.py) with
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True
```
to relax the attention kernel setting and use other kernels than Flash Attention.
</details>

<details>
<summary>
I got `Error compiling objects for extension`
</summary>
<br/>

You may see error log of:
> unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

This is probably because your versions of CUDA and Visual Studio are incompatible. (see also https://stackoverflow.com/questions/78515942/cuda-compatibility-with-visual-studio-2022-version-17-10 for a discussion in stackoverflow).<br> 
You may be able to fix this by adding the `-allow-unsupported-compiler` argument to `nvcc` after L48 in the [setup.py](https://github.com/facebookresearch/sam2/blob/main/setup.py). <br>
After adding the argument, `get_extension()` will look like this:
```python
def get_extensions():
    srcs = ["sam2/csrc/connected_components.cu"]
    compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-allow-unsupported-compiler"  # Add this argument
        ],
    }
    ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    return ext_modules
```
</details>
