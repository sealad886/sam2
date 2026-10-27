# Contributing to segment-anything
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints, using the `ufmt format` command. Linting requires `black==24.2.0`, `usort==1.0.2`, and `ufmt==2.0.0b2`, which can be installed via `pip install -e ".[dev]"`.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Inference regression checks

Install the project and test/formatting tools in a virtual environment:

```bash
python -m pip install -e . pytest 'av>=13.0.0' black==24.2.0 usort==1.0.2 ufmt==2.0.0b2
python -m pytest -q tests
python -m ufmt check sam2 tools tests setup.py
```

Tests that need MPS or CUDA skip when that hardware is unavailable. Run them
directly on an Apple Silicon Mac to exercise MPS; CPU CI does not qualify GPU
execution. Packaging tests cover Mac defaults and explicit CUDA build overrides
without requiring a CUDA toolkit. The real-checkpoint qualification command in
[INSTALL.md](INSTALL.md#apple-silicon-macos) additionally checks image, automatic
masks, two-object video, reset/reuse, mask agreement and retained memory.

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to segment-anything, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
