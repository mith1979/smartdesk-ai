#!/usr/bin/env python3
"""
File: scripts/test_create_ticket.py

Purpose:
    Verify live GitHub ticket creation after explicit terminal confirmation.
Input:
    GITHUB_TOKEN and GITHUB_REPOSITORY from environment variables,
    project .env, or shared fde_program/.env.
    Reuses path constants from scripts/verify_github.py.
Output:
    Actual GitHub issue number, URL, and initial state.
FDE connection:
    External tool integration (Week 7 concept) and human confirmation
    before a write. Later the LangGraph workflow will manage this approval.
Side effects:
    Typing CREATE submits one real issue with fictional test content.
    No LLM calls, local database writes, automatic retries, or ticket closure.
Limitations:
    Development check only: no employee-email mapping or category label yet.
    Each successful run creates a new issue. Run once and keep its number.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from verify_github import PROJECT_DIR, FDE_DIR, EXPECTED_REPOSITORY


def main():
    """Preview a fixed test ticket and create it only after confirmation."""
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repository = os.getenv("GITHUB_REPOSITORY", "").strip()

    if not token or repository != EXPECTED_REPOSITORY:
        print("Check configuration by running scripts/verify_github.py.")
        return 1

    # Fixed, fictional data makes the first write easy to review.
    title = "[SmartDesk test] Monitor flickering"
    body = (
        "## Issue description\n"
        "Fictional employee reports a monitor flickering for two days.\n"
        "The demonstration KB does not define a replacement procedure.\n\n"
        "## Category\nIT\n\n"
        "## Test marker\n"
        "SmartDesk integration test. Not a real support request.\n"
    )

    print(f"Repository: {repository}")
    print(f"Title: {title}\n\n{body}")
    try:
        confirmation = input("Type CREATE to submit this real test issue: ")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. No request sent.")
        return 0
    if confirmation.strip() != "CREATE":
        print("Cancelled. No request sent.")
        return 0

    request = Request(
        f"https://api.github.com/repos/{repository}/issues",
        data=json.dumps({"title": title, "body": body}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "SmartDesk-Capstone",
        },
        method="POST",
    )

    # Do not automatically retry writes: a lost response can hide success.
    try:
        with urlopen(request, timeout=20) as response:
            ticket = json.load(response)
    except HTTPError as error:
        print(f"GitHub returned HTTP {error.code}.")
        if error.code in (401, 403, 404):
            print("Check token expiry, repository access, and Issues write permission.")
        elif error.code == 422:
            print("GitHub rejected the ticket data or request frequency.")
        print("Inspect repository Issues before retrying; no automatic retry occurred.")
        return 1
    except (URLError, TimeoutError, OSError, ValueError):
        print("Could not confirm ticket creation from GitHub's response.")
        print("Check repository Issues before retrying: the ticket may exist.")
        return 1

    if not isinstance(ticket, dict) or not {"number", "html_url", "state"} <= ticket.keys():
        print("Unexpected response. Inspect repository Issues before retrying.")
        return 1

    print(f"\nCreated ticket: #{ticket['number']}")
    print(f"URL: {ticket['html_url']}")
    print(f"Initial state: {ticket['state']}")
    print("Ticket-write access verified. Keep this issue for the status-read test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
