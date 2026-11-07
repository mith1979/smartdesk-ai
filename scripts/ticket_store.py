#!/usr/bin/env python3
"""
File: scripts/ticket_store.py

Purpose:
    Persist employee-email to GitHub-ticket associations in local SQLite.
    Functions are reusable by the later ticket creation and LangGraph flows.
Input:
    CLI: link EMAIL ISSUE_NUMBER, or list EMAIL.
Output:
    Ticket references associated with an email, or an explicit empty result.
FDE connection:
    Session state remembers the ongoing conversation; this persistent mapping
    supports external-tool lookup after the Python process exits.
    It complements the state and tool concepts discussed in Weeks 4 and 7.
Side effects:
    Creates storage/tickets.sqlite3 and its table on first use.
    link writes a local association; list reads associations.
    No GitHub requests, external changes, or LLM calls.
Limitations:
    Stores references only, never cached ticket status or comments.
    Manual link assumes the issue number has already been verified.
    Email is a demonstration identifier, not proof of employee identity.
    Protect this local database and exclude it from version control.
"""

import argparse
import re
import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_DIR / "storage" / "tickets.sqlite3"
REPOSITORY = "mith1979/smartdesk-tickets"


def normalize_email(email):
    """Apply basic input validation and consistent demo email matching."""
    email = email.strip().lower()
    if len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise ValueError("Enter an email such as employee@example.com")
    return email


def connect_database(database_file=DATABASE_FILE):
    """Initialize local storage; callers close the returned connection."""
    database_file = Path(database_file)
    database_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_file, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS employee_tickets ("
            "repository TEXT NOT NULL, "
            "issue_number INTEGER NOT NULL CHECK(issue_number > 0), "
            "email TEXT NOT NULL, "
            "linked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "PRIMARY KEY(repository, issue_number))"
        )
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection


def link_ticket(email, issue_number, database_file=DATABASE_FILE):
    """Link a verified ticket; repeat links are safe, reassignment is rejected."""
    email = normalize_email(email)
    if type(issue_number) is not int or issue_number < 1:
        raise ValueError("Issue number must be a positive integer")
    connection = connect_database(database_file)
    try:
        with connection:
            connection.execute(
                "INSERT INTO employee_tickets(repository, issue_number, email) "
                "VALUES (?, ?, ?) ON CONFLICT(repository, issue_number) DO NOTHING",
                (REPOSITORY, issue_number, email),
            )
            owner = connection.execute(
                "SELECT email FROM employee_tickets "
                "WHERE repository = ? AND issue_number = ?",
                (REPOSITORY, issue_number),
            ).fetchone()
            if owner["email"] != email:
                raise ValueError("This ticket is already linked to another email")
    finally:
        connection.close()


def find_tickets(email, database_file=DATABASE_FILE):
    """Return references only; the API layer fetches live details separately."""
    email = normalize_email(email)
    connection = connect_database(database_file)
    try:
        rows = connection.execute(
            "SELECT repository, issue_number FROM employee_tickets "
            "WHERE email = ? AND repository = ? ORDER BY issue_number DESC",
            (email, REPOSITORY),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description="Manage local employee-ticket links")
    commands = parser.add_subparsers(dest="command", required=True)
    link = commands.add_parser("link", help="Associate an already verified ticket")
    link.add_argument("email")
    link.add_argument("number", type=int)
    listing = commands.add_parser("list", help="Find tickets associated with an email")
    listing.add_argument("email")
    args = parser.parse_args()
    try:
        if args.command == "link":
            link_ticket(args.email, args.number)
            print(f"Linked ticket #{args.number} to {normalize_email(args.email)}")
            print("Local mapping saved. GitHub was not changed.")
        else:
            tickets = find_tickets(args.email)
            if not tickets:
                print("No locally associated tickets found for this email.")
            for ticket in tickets:
                print(f"{ticket['repository']} — #{ticket['issue_number']}")
    except (ValueError, sqlite3.Error, OSError) as error:
        print(f"Mapping operation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
