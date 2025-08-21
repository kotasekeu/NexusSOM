# Installation Guide

The recommended and primary method for installing and running NexusSom is via Docker. This ensures a consistent and isolated environment for all dependencies and application components. However, for advanced users or specific deployment scenarios, manual installation on Ubuntu is also supported.

## Prerequisites

- Git (required for both installation methods)
- For Docker installation:
  - [Docker](https://www.docker.com/) (version 20.10 or newer recommended)
  - [Docker Compose](https://docs.docker.com/compose/) (if not included in your Docker installation)
  - Optionally, [PyCharm](https://www.jetbrains.com/pycharm/) for integrated development and Docker support

- For native Ubuntu installation (without Docker):
  - Ubuntu 22.04 or newer recommended
  - [Python 3.12](https://www.python.org/downloads/)
  - [pip](https://pip.pypa.io/en/stable/)
  - [venv](https://docs.python.org/3/library/venv.html) module for virtual environments

## Installation via Command Line

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-org/NexusSom.git
   cd NexusSom
   ```

2. **Build and Start the Application**

   Execute the following command in the project root directory:

   ```bash
   docker compose up --build
   ```

   This will build the Docker image and start the application in a containerized environment.

## Installation via PyCharm

1. Open the project folder in PyCharm.
2. Ensure Docker is configured in your IDE's settings.
3. Locate the `docker-compose.yml` file in the project root.
4. Right-click on `docker-compose.yml` and select **Compose Up** or use the Docker tool window to start the service.

The application will be built and started automatically in a Docker container.

## Installation without Docker (Ubuntu)

If you prefer to install and run NexusSom without Docker, follow these steps on Ubuntu:

1. **Install Python 3.12 and pip**

   ```bash
   sudo apt update
   sudo apt install python3.12 python3.12-venv python3-pip
   ```

2. **Clone the Repository**

   ```bash
   git clone https://github.com/your-org/NexusSom.git
   cd NexusSom
   ```

3. **Create and Activate a Virtual Environment**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r app/requirements.txt
   ```

5. **Run the Application**

   ```bash
   cd app
   python3 main.py
   ```

All required libraries will be installed as specified in `app/requirements.txt`. No additional dependencies are needed.

## Python Dependencies

All required Python libraries are specified in `app/requirements.txt` and are installed automatically during the Docker build process. The main libraries include:

- **passlib**: Password hashing utilities
- **pydantic**: Data validation and settings management
- **pytest**: Testing framework
- **httpx**: HTTP client for Python
- **pandas**: Data analysis and manipulation
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing
- **psutil**: System and process utilities
- **tqdm**: Progress bars for Python loops

No manual installation of Python packages is required; all dependencies are managed within the Docker container.

## Notes

- The application runs entirely within Docker; no Python environment setup is required on the host machine.
- For troubleshooting, consult the Docker logs using `docker compose logs`.

For further information, please refer to the project documentation or contact the maintainers.
