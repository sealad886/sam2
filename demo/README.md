# SAM 2 Demo

Welcome to the SAM 2 Demo! This project consists of a frontend built with React TypeScript and Vite and a backend service using Python Flask and Strawberry GraphQL. Both components can be run in Docker containers or locally on MPS (Metal Performance Shaders) or CPU. However, running the backend service on MPS or CPU devices may result in significantly slower performance (FPS).

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Docker and Docker Compose
- [OPTIONAL] Node.js and Yarn for running frontend locally
- [OPTIONAL] Anaconda for running backend locally

### Installing Docker

To install Docker, follow these steps:

1. Go to the [Docker website](https://www.docker.com/get-started)
2. Follow the installation instructions for your operating system.

### [OPTIONAL] Installing Node.js and Yarn

Use Node.js 24 LTS (24.22.2 or newer) and Yarn Classic 1.22.22, pinned by
`demo/frontend/package.json`. Install Node.js using the
[official instructions](https://nodejs.org/en/download/), then run Yarn through
Corepack from the frontend directory:

```bash
corepack yarn install --frozen-lockfile
```

If your Node.js distribution does not include Corepack, use
`npm exec --yes --package=yarn@1.22.22 -- yarn install --frozen-lockfile`.
The frontend uses the security-supported Vite 6.4 release line.

### [OPTIONAL] Installing Anaconda

To install Anaconda, follow these steps:

1. Go to the [Anaconda website](https://www.anaconda.com/products/distribution).
2. Follow the installation instructions for your operating system.

## Quick Start

To get both the frontend and backend running quickly using Docker, you can use the following command:

```bash
docker compose up --build
```

> [!WARNING]
> On macOS, Docker containers only support running on CPU. MPS is not supported through Docker. If you want to run the demo backend service on MPS, you will need to run it locally (see "Running the Backend Locally" below).

This will build and start both services. You can access them at:

- **Frontend:** [http://localhost:7262](http://localhost:7262)
- **Backend:** [http://localhost:7263/graphql](http://localhost:7263/graphql)

## Running Backend with MPS Support

MPS (Metal Performance Shaders) is not supported with Docker. To use MPS, you need to run the backend on your local machine.

### Setting Up Your Environment

1. **Create Conda environment**

   Create a new Conda environment for this project by running the following command or use your existing conda environment for SAM 2:

   ```
   conda create --name sam2-demo python=3.10 --yes
   ```

   This will create a new environment named `sam2-demo` with Python 3.10 as the interpreter.

2. **Activate the Conda environment:**

   ```bash
   conda activate sam2-demo
   ```

3. **Install ffmpeg**

   ```bash
   conda install -c conda-forge ffmpeg
   ```

4. **Install SAM 2 demo dependencies:**

Install project dependencies by running the following command in the SAM 2 checkout root directory:

```bash
pip install -e '.[interactive-demo]'
```

### Running the Backend Locally

Download the SAM 2 checkpoints:

```bash
(cd ./checkpoints && ./download_ckpts.sh)
```

Use the following command to start the backend with MPS support:

```bash
cd demo/backend/server/
```

```bash
APP_ROOT="$(pwd)/../../../" \
API_URL=http://localhost:7263 \
MODEL_SIZE=base_plus \
DATA_PATH="$(pwd)/../../data" \
DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4 \
gunicorn \
    --worker-class gthread app:app \
    --workers 1 \
    --threads 2 \
    --bind 0.0.0.0:7263 \
    --timeout 60
```

Options for the `MODEL_SIZE` argument are "tiny", "small", "base_plus" (default), and "large".

> [!WARNING]
> Running the backend service on MPS devices can cause fatal crashes with the Gunicorn worker due to insufficient MPS memory. Try switching to CPU devices by setting the `SAM2_DEMO_FORCE_CPU_DEVICE=1` environment variable.

### Starting the Frontend

If you wish to run the frontend separately (useful for development), follow these steps:

1. **Navigate to demo frontend directory:**

   ```bash
   cd demo/frontend
   ```

2. **Install dependencies:**

   ```bash
   corepack yarn install --frozen-lockfile
   ```

3. **Start the development server:**

   ```bash
   corepack yarn dev --port 7262
   ```

This will start the frontend development server on [http://localhost:7262](http://localhost:7262).

### Frontend validation and dependency maintenance

Run `corepack yarn relay`, `corepack yarn lint`, `corepack yarn build`, and
`corepack yarn audit --groups dependencies` before submitting frontend changes.
The production dependency audit must pass. The refresh consolidates upstream
Dependabot PRs [#755](https://github.com/facebookresearch/sam2/pull/755),
[#754](https://github.com/facebookresearch/sam2/pull/754),
[#752](https://github.com/facebookresearch/sam2/pull/752),
[#750](https://github.com/facebookresearch/sam2/pull/750),
[#745](https://github.com/facebookresearch/sam2/pull/745),
[#744](https://github.com/facebookresearch/sam2/pull/744), and
[#743](https://github.com/facebookresearch/sam2/pull/743); their original fixes
and package maintainers are credited in the validation receipt.

The scoped Yarn resolutions have specific owners. Nx's `semver`,
typescript-eslint's `minimatch`, and Tailwind Typography's
`postcss-selector-parser` use patched versions within their existing major
release because their parents pin vulnerable versions. React Relay's wildcard
type dependencies are aligned with this application's React 18 and declared
Relay type versions. The import resolver, import plugin, React Refresh plugin,
and their two `eslint-module-utils` versions retain the previously working lint
toolchain; newer combinations change StyleX import diagnostics or fail inside
the import plugin. These constraints keep the existing lint rules enabled.

The full development dependency audit still reports
[GHSA-g3hv-75wg-m249](https://github.com/advisories/GHSA-g3hv-75wg-m249) in the
lint toolchain's `minimatch` versions. Its fix requires major version 11;
upgrading ESLint/typescript-eslint is a separate toolchain migration. These
packages do not enter the production dependency graph. See the
[dependency validation receipt](../docs/validation/frontend-maintenance-2026-11-03.json)
for the measured audit counts, version choices, and validation limits.

## Docker Tips

- To rebuild the Docker containers (useful if you've made changes to the Dockerfile or dependencies):

  ```bash
  docker compose up --build
  ```

- To stop the Docker containers:

  ```bash
  docker compose down
  ```

## Contributing

Contributions are welcome! Please read our contributing guidelines to get started.

## License

See the LICENSE file for details.

---

By following these instructions, you should have a fully functional development environment for both the frontend and backend of the SAM 2 Demo. Happy coding!
