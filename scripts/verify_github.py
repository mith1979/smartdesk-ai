#!/usr/bin/env python3
"""
File: scripts/verify_github.py

Purpose:
    Verify authenticated access to SmartDesk's GitHub ticket repository.

Input:
    GITHUB_TOKEN and GITHUB_REPOSITORY from environment variables,
    the project .env, or the shared fde_program/.env, in that order.
    Save this file inside smartdesk_capstone/scripts/.

Output:
    Repository name, Issues availability, and a read-access result.
    Exit code 0 for success; 1 for configuration or API failure.

FDE connection:
    Week 7: External API integration before exposing an agent tool.
    Week 8: Explicit failure handling without exposing credentials.
    This is a standalone connectivity check, not a LangGraph node yet.

Side effects:
    One authenticated GET request to GitHub. No tickets or files created.
    Does not call an LLM, rebuild the index, or print the token.

Limitations:
    Verifies repository read access and Issues availability only.
    Ticket-write permission is verified later with an actual creation test.
"""

import json
import os
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


# Resolve paths from this script, independent of the terminal's directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Use the shared FDE .env when running inside the course workspace.
# In a standalone clone, fall back to the project's own .env directory.
FDE_DIR = next(
    (parent for parent in PROJECT_DIR.parents if parent.name == "fde_program"),
    PROJECT_DIR,
)
# FDE_DIR = PROJECT_DIR.parents[2]
EXPECTED_REPOSITORY = "mith1979/smartdesk-tickets"


def main():
    """Load configuration and perform a read-only repository check."""
    # Earlier values win: shell environment > project .env > shared .env.
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)

    token = os.getenv("GITHUB_TOKEN", "").strip()
    repository = os.getenv("GITHUB_REPOSITORY", "").strip()

    missing = []
    if not token:
        missing.append("GITHUB_TOKEN")
    if not repository:
        missing.append("GITHUB_REPOSITORY")

    if missing:
        print("Missing configuration: " + ", ".join(missing))
        print("Checked configuration files:")
        for path in (PROJECT_DIR / ".env", FDE_DIR / ".env"):
            status = "exists" if path.is_file() else "not found"
            print(f"  {path} ({status})")
        print("Add the missing values without sharing your token.")
        return 1

    # Keep this development check restricted to the intended repository.
    if repository != EXPECTED_REPOSITORY:
        print(f"Set GITHUB_REPOSITORY={EXPECTED_REPOSITORY}")
        print("Check existing shell variables and both .env files.")
        return 1

    request = Request(
        f"https://api.github.com/repos/{repository}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "SmartDesk-Capstone",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            data = json.load(response)
    except HTTPError as error:
        hints = {
            401: "Token is invalid or expired.",
            403: "Check token permissions, approval status, or rate limits.",
            404: "Check the repository name and token repository access.",
            429: "GitHub rate limit reached. Try again later.",
        }
        print(f"GitHub HTTP {error.code}: " + hints.get(
            error.code, "Repository access check failed. Try again later."
        ))
        return 1
    except (URLError, TimeoutError, socket.timeout):
        print("Could not reach GitHub. Check connectivity and try again.")
        return 1
    except (ValueError, OSError):
        print("Could not read GitHub's response. Try again later.")
        return 1

    if not isinstance(data, dict) or not {"full_name", "has_issues"} <= data.keys():
        print("GitHub returned an unexpected repository response.")
        return 1

    print(f"Repository: {data['full_name']}")
    print(f"Issues enabled: {data['has_issues']}")

    if not data["has_issues"]:
        print("Enable Issues in repository Settings > General > Features.")
        return 1

    print("Read access verified. No ticket created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
