#!/usr/bin/env python3
"""
File: scripts/tickets.py

Purpose:
    Provide reusable ticket creation and employee-email status lookup.
    The CLI previews a ticket and requires CREATE before an external write.
Input:
    create: email, summary, description, category; terminal confirmation.
    status: employee email. Credentials use existing .env loading rules.
Output:
    Real ticket reference, or live ticket states and recent comments.
FDE connection:
    Week 7 external-tool concepts; later graph nodes will call these functions.
    Weeks 4-5 conversation state and confirmation will surround these tools.
Side effects:
    create_ticket() writes one GitHub issue only when confirmed=True, then
    records its employee association in storage/tickets.sqlite3.
    Status calls read GitHub; local storage initializes if absent.
    No LLM calls and no automatic retries of writes.
Dependencies:
    Keep verify_github.py, check_ticket.py, and ticket_store.py in scripts/.
Limitations:
    Explicit confirmation is a caller contract, not employee authentication.
    CLI confirmation is implemented here; LangGraph approval comes next.
    Category is recorded in the body, not yet a GitHub label.
    A network failure after submission can leave creation outcome uncertain.
"""

import argparse
import json
import os
import sqlite3
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from verify_github import PROJECT_DIR, FDE_DIR, EXPECTED_REPOSITORY
from check_ticket import get_ticket, get_recent_comments
from ticket_store import normalize_email, link_ticket, find_tickets


class TicketError(Exception):
    """An operational failure that can be shown without leaking credentials."""


def load_token():
    """Load existing environment > project .env > shared FDE .env."""
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(FDE_DIR / ".env", override=False)
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token or os.getenv("GITHUB_REPOSITORY", "").strip() != EXPECTED_REPOSITORY:
        raise TicketError("Check GitHub configuration with scripts/verify_github.py.")
    return token


def validate_draft(email, summary, description, category):
    """Require all mandatory fields before attempting an external write."""
    email = normalize_email(email)
    summary, description, category = summary.strip(), description.strip(), category.upper().strip()
    if not summary or len(summary) > 200:
        raise ValueError("Summary must contain 1 to 200 characters.")
    if not description or len(description) > 10000:
        raise ValueError("Description must contain 1 to 10000 characters.")
    if category not in ("IT", "HR"):
        raise ValueError("Category must be IT or HR.")
    return email, summary, description, category


def create_ticket(email, summary, description, category, *, confirmed=False):
    """Create an approved draft and map its returned number to the employee."""
    if confirmed is not True:
        raise ValueError("Explicit confirmation is required before ticket creation.")
    email, summary, description, category = validate_draft(
        email, summary, description, category
    )
    token = load_token()
    # Ensure local storage is accessible before sending the write request.
    find_tickets(email)
    body = f"## Issue description\n{description}\n\n## Category\n{category}\n"
    request = Request(
        f"https://api.github.com/repos/{EXPECTED_REPOSITORY}/issues",
        data=json.dumps({"title": summary, "body": body}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "SmartDesk-Capstone",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            ticket = json.load(response)
    except HTTPError as error:
        raise TicketError(
            f"GitHub returned HTTP {error.code}. Check repository Issues before retrying."
        ) from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise TicketError(
            "Creation outcome could not be confirmed. Check GitHub Issues before retrying; "
            "a ticket may already exist."
        ) from None
    if not isinstance(ticket, dict) or not {"number", "html_url", "state"} <= ticket.keys():
        raise TicketError("Unexpected creation response. Check GitHub Issues before retrying.")
    try:
        link_ticket(email, ticket["number"])
    except (sqlite3.Error, OSError, ValueError):
        # Preserve the successful external result so callers do not create again.
        return {
            **ticket,
            "mapping_saved": False,
            "warning": "Ticket created, but email mapping failed. Repair the local link; do not recreate.",
        }
    return {**ticket, "mapping_saved": True, "warning": None}


def list_employee_tickets(email):
    """Return live details with per-ticket errors, preserving partial results."""
    references = find_tickets(email)
    if not references:
        return []
    token = load_token()
    results = []
    for reference in references:
        number = reference["issue_number"]
        try:
            ticket = get_ticket(number, token)
        except HTTPError as error:
            results.append({"number": number, "error": f"GitHub HTTP {error.code}"})
            continue
        except (URLError, TimeoutError, OSError, ValueError):
            results.append({"number": number, "error": "Could not fetch live ticket details"})
            continue
        try:
            comments = get_recent_comments(number, token)
            comments_error = None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            comments = []
            comments_error = "Comments unavailable; this does not mean there are no comments."
        results.append({
            "number": number, "title": ticket["title"], "state": ticket["state"],
            "url": ticket["html_url"], "comments": comments,
            "comments_error": comments_error, "error": None,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="SmartDesk live ticket tools")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--email", required=True)
    create = commands.add_parser("create")
    create.add_argument("--email", required=True)
    create.add_argument("--summary", required=True)
    create.add_argument("--description", required=True)
    create.add_argument("--category", choices=["IT", "HR"], required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            draft = validate_draft(args.email, args.summary, args.description, args.category)
            for name, value in zip(("Email", "Summary", "Description", "Category"), draft):
                print(f"{name}: {value}")
            if input("Type CREATE to submit, or anything else to cancel: ").strip() != "CREATE":
                print("Cancelled. No ticket created.")
                return 0
            ticket = create_ticket(*draft, confirmed=True)
            print(f"Created #{ticket['number']}: {ticket['html_url']}")
            print(f"Initial state: {ticket['state']}")
            if ticket["warning"]:
                print(ticket["warning"])
                print(f"Repair with: python scripts/ticket_store.py link {draft[0]} {ticket['number']}")
        else:
            tickets = list_employee_tickets(args.email)
            if not tickets:
                print("No locally associated tickets found for this email.")
            for ticket in tickets:
                if ticket["error"]:
                    print(f"#{ticket['number']}: {ticket['error']}")
                    continue
                print(f"\n#{ticket['number']}: {ticket['title']} — {ticket['state']}")
                print(ticket["url"])
                if ticket["comments_error"]:
                    print(ticket["comments_error"])
                elif not ticket["comments"]:
                    print("No comments yet.")
                for comment in ticket["comments"]:
                    print(f"  {comment['user']['login']}: {comment['body']}")
            if any(t["error"] or t.get("comments_error") for t in tickets):
                return 1
    except (TicketError, ValueError, sqlite3.Error, OSError) as error:
        print(f"Ticket operation failed: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
