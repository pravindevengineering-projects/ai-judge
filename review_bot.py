"""AI PR reviewer.

Reads a pull request's diff via PyGithub, filters out noise files, asks Claude
Sonnet 5 to find logic bugs (returning one-click GitHub ``suggestion`` fixes),
and posts the findings back as inline review comments with review state COMMENT.

Designed to run inside GitHub Actions on a ``pull_request`` event, where
GITHUB_REPOSITORY and GITHUB_EVENT_PATH are set automatically.
"""

from __future__ import annotations

import json
import os
import re
import sys

import anthropic
from github import Auth, Github

MODEL = "claude-sonnet-5"

# Files whose diffs are noise for logic-bug review.
_MINIFIED_RE = re.compile(r"\.min\.(js|css)$", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are a meticulous code reviewer. You are given the diff of a GitHub pull \
request. Your only job is to find genuine LOGIC BUGS introduced by the diff: \
off-by-one errors, inverted conditions, incorrect operators, mishandled \
edge cases, wrong variable usage, broken control flow, resource/None handling \
mistakes, and similar defects that would cause incorrect behavior.

Strict rules:
- Report ONLY real logic bugs. Do NOT report style, formatting, naming, or \
subjective preferences.
- Every finding MUST propose a concrete fix using GitHub's suggestion syntax: \
a fenced code block whose opening fence is exactly ```suggestion, containing \
the corrected replacement for the commented line(s), then a closing ```.
- Anchor each finding to a specific changed line that actually appears in the \
diff. Use the file path exactly as shown in the diff header, and the line \
number in the NEW version of the file (the right side of the diff).
- If you find no logic bugs, return an empty "findings" list.

Return your answer as JSON matching the provided schema."""

# Structured-output schema: guarantees valid, parseable JSON.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative file path, exactly as in the diff.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number in the new version of the file (RIGHT side).",
                    },
                    "body": {
                        "type": "string",
                        "description": "Markdown comment containing a ```suggestion fix block.",
                    },
                },
                "required": ["path", "line", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def is_reviewable(filename: str) -> bool:
    """Return False for lockfiles, CSVs, and minified assets."""
    if filename == "package-lock.json" or filename.endswith("/package-lock.json"):
        return False
    if filename.endswith(".csv"):
        return False
    if _MINIFIED_RE.search(filename):
        return False
    return True


def collect_diff(pull) -> tuple[str, set[str]]:
    """Build a single diff blob from reviewable files.

    Returns the blob and the set of reviewed file paths.
    """
    parts: list[str] = []
    reviewed: set[str] = set()
    for f in pull.get_files():
        if not is_reviewable(f.filename):
            continue
        if not f.patch:  # binary or too-large: no textual patch
            continue
        reviewed.add(f.filename)
        parts.append(f"--- {f.filename} ---\n{f.patch}")
    return "\n\n".join(parts), reviewed


def find_bugs(diff_blob: str) -> list[dict]:
    """Ask Claude for logic bugs; return the parsed findings list."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": f"Review this pull request diff for logic bugs:\n\n{diff_blob}",
            }
        ],
    )
    if response.stop_reason == "max_tokens":
        print("Warning: response was truncated (max_tokens reached); findings may be incomplete.")
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        return []
    return json.loads(text_block.text).get("findings", [])


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")

    repo_name = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not repo_name or not event_path:
        sys.exit("GITHUB_REPOSITORY and GITHUB_EVENT_PATH must be set (run inside Actions)")

    with open(event_path, encoding="utf-8") as fh:
        event = json.load(fh)
    pr_number = event.get("pull_request", {}).get("number") or event.get("number")
    if not pr_number:
        sys.exit("Could not determine the pull request number from the event payload")

    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo(repo_name)
    pull = repo.get_pull(int(pr_number))

    diff_blob, reviewed = collect_diff(pull)
    if not diff_blob:
        print("No reviewable files in this PR; nothing to do.")
        return 0

    findings = find_bugs(diff_blob)
    if not findings:
        print("No logic bugs found.")
        return 0

    comments = [
        {"path": f["path"], "line": f["line"], "side": "RIGHT", "body": f["body"]}
        for f in findings
        if f.get("path") in reviewed and f.get("line") and f.get("body")
    ]
    if not comments:
        print("Findings did not map to reviewed lines; nothing to post.")
        return 0

    pull.create_review(
        body=f"🤖 AI review found {len(comments)} potential logic bug(s).",
        event="COMMENT",
        comments=comments,
    )
    print(f"Posted {len(comments)} inline comment(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
