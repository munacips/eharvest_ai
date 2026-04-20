import os
from pathlib import Path

from dotenv import load_dotenv


def pytest_sessionstart(session):
    if os.getenv("RUN_PLATFORM_TESTS") == "1":
        return
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
