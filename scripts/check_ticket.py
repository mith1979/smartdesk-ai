#!/usr/bin/env python3
"""
File: scripts/check_ticket.py

Purpose:
    Fetch one GitHub ticket's live status and recent support comments.
Input:
    Issue number on the command line; GitHub configuration from the
    environment, project .env, or shared FDE .env, in that order.
    Keep verify_github.py in the same scripts directory.
Output:
    Issue number, title, live state, URL, and up to five recent comments.
FDE connection:
    Week 7 concept: read operations against an external API.
    Later this function can be reused by the LangGraph status workflow.
Side effects:
    Read-only GitHub requests. No ticket edits, local writes, or LLM calls.
Limitations:
    Looks up a supplied issue number, not an employee email yet.
    Comment history is paginated; only the last five comments are displayed.
    GitHub open/closed state is reported without inventing workflow stages.
"""

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from verify_github import PROJECT_DIR, FDE_DIR, EXPECTED_REPOSITORY


def read_json(path, token):
    """Read a fixed GitHub API path without logging credentials."""
    request = Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "SmartDesk-Capstone",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def get_ticket(number, token):
    """Return live ticket data, excluding pull requests."""
    base = f"/repos/{EXPECTED_REPOSITORY}/issues/{number}"
    ticket = read_json(base, token)
    if "pull_request" in ticket:
        raise ValueError("This number identifies a pull request, not a support ticket.")
    return ticket


def get_recent_comments(number, token):
    """Read all comment pages and retain only the five most recent entries."""
    recent = []
    page = 1
    while True:
        comments = read_json(
            f"/repos/{EXPECTED_REPOSITORY}/issues/{number}/comments"
            f"?per_page=100&page={page}", token,
        )
        recent = (recent + comments)[-5:]
        if len(comments) < 100:
            return recent
        page += 1


def main():
    parser = argparse.ArgumentParser(description="Read a live SmartDesk ticket")
    parser.add_argument("number", type=int, help="GitHub issue number, such as 1")
    args = parser.parse_args()
    if args.number < 1:
        parser.error("Issue number must be positive")

    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repository = os.getenv("GITHUB_REPOSITORY", "").strip()
    if not token or repository != EXPECTED_REPOSITORY:
        print("Check configuration with scripts/verify_github.py.")
        return 1

    try:
        ticket = get_ticket(args.number, token)
    except HTTPError as error:
        print(f"Ticket lookup failed: GitHub HTTP {error.code}.")
        if error.code == 404:
            print("Ticket was not found or this token cannot access it.")
        return 1
    except (URLError, TimeoutError, OSError):
        print("Could not reach GitHub. Try again later.")
        return 1
    except ValueError:
        print("The response was invalid or this number identifies a pull request.")
        return 1

    print(f"Ticket: #{ticket['number']}")
    print(f"Title: {ticket['title']}")
    print(f"Live state: {ticket['state']}")
    print(f"Updated: {ticket['updated_at']}")
    print(f"URL: {ticket['html_url']}")

    # A comment API failure must not be reported as 'no comments'.
    try:
        comments = get_recent_comments(args.number, token)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        print("\nStatus retrieved, but comments could not be fetched. Try again later.")
        return 1

    print("\nRecent comments:")
    if not comments:
        print("No comments yet.")
    for comment in comments:
        print(f"\n{comment['user']['login']} — {comment['created_at']}")
        print(comment['body'])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
