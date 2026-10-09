"""AI PR reviewer.

Reads a pull request's diff via PyGithub, filters out noise files, asks Claude
via AWS Bedrock to find logic bugs (returning one-click GitHub ``suggestion``
fixes), and posts the findings back as inline review comments.

Phase 1 reliability improvements:
- Line-number validation: only posts comments on lines that actually appear in
  the diff, eliminating 422 errors from GitHub.
- Per-file chunking: reviews each file independently so large PRs never exceed
  token limits.
- Deduplication: skips posting if the bot already reviewed this exact commit.
- 422 fallback: if GitHub still rejects an inline comment, falls back to a
  PR-level comment so no finding is silently lost.

Designed to run inside GitHub Actions on a ``pull_request`` event, where
GITHUB_REPOSITORY and GITHUB_EVENT_PATH are set automatically.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

import anthropic
from github import Auth, Github
from github.GithubException import GithubException

MODEL = "us.anthropic.claude-sonnet-4-6"  # us-west-2 cross-region inference

# Max diff characters sent per Claude call. Keeps well inside token limits.
MAX_CHUNK_CHARS = 12_000

# Files whose diffs are noise for logic-bug review.
_MINIFIED_RE = re.compile(r"\.min\.(js|css)$", re.IGNORECASE)

BOT_MARKER = "<!-- ai-judge-review -->"

SYSTEM_PROMPT = """\
You are a meticulous code reviewer. You are given the diff of one file from a \
GitHub pull request. Your only job is to find genuine LOGIC BUGS introduced by \
the diff: off-by-one errors, inverted conditions, incorrect operators, \
mishandled edge cases, wrong variable usage, broken control flow, \
resource/None handling mistakes, and similar defects that would cause \
incorrect behavior.

Strict rules:
- Report ONLY real logic bugs. Do NOT report style, formatting, naming, or \
subjective preferences.
- Every finding MUST propose a concrete fix using GitHub's suggestion syntax: \
a fenced code block whose opening fence is exactly ```suggestion, containing \
the corrected replacement for the commented line(s), then a closing ```.
- Anchor each finding to a specific ADDED line (starting with '+') that \
actually appears in the diff. Use the file path exactly as shown, and the \
line number in the NEW version of the file (right side of the diff).
- If you find no logic bugs, return an empty "findings" list.

Return your answer as JSON matching the provided schema."""

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


@dataclass
class FileDiff:
    filename: str
    patch: str
    valid_lines: set[int]  # line numbers that appear in the diff (RIGHT side)


def is_reviewable(filename: str) -> bool:
    if filename == "package-lock.json" or filename.endswith("/package-lock.json"):
        return False
    if filename.endswith(".csv"):
        return False
    if _MINIFIED_RE.search(filename):
        return False
    return True


def parse_valid_lines(patch: str) -> set[int]:
    """Extract new-file line numbers that appear in the patch."""
    valid: set[int] = set()
    new_line = 0
    for raw in patch.splitlines():
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if hunk:
            new_line = int(hunk.group(1))
            continue
        if raw.startswith("-"):
            continue  # deleted line — no new-file number
        if raw.startswith("+"):
            valid.add(new_line)
        new_line += 1
    return valid


def collect_files(pull) -> list[FileDiff]:
    """Return FileDiff objects for every reviewable file in the PR."""
    files: list[FileDiff] = []
    for f in pull.get_files():
        if not is_reviewable(f.filename):
            continue
        if not f.patch:
            continue
        files.append(FileDiff(
            filename=f.filename,
            patch=f.patch,
            valid_lines=parse_valid_lines(f.patch),
        ))
    return files


def already_reviewed(pull, commit_sha: str) -> bool:
    """Return True if the bot already posted a review for this commit."""
    for review in pull.get_reviews():
        if (
            review.user.login.endswith("[bot]")
            and review.commit_id == commit_sha
            and BOT_MARKER in (review.body or "")
        ):
            return True
    return False


@dataclass
class FileResult:
    findings: list[dict]
    input_tokens: int
    output_tokens: int


def find_bugs_for_file(client: anthropic.AnthropicBedrock, fd: FileDiff) -> FileResult:
    """Run one Claude call for a single file's diff."""
    patch = fd.patch
    if len(patch) > MAX_CHUNK_CHARS:
        patch = patch[:MAX_CHUNK_CHARS]
        print(f"  {fd.filename}: diff truncated to {MAX_CHUNK_CHARS} chars")

    diff_text = f"File: {fd.filename}\n\n{patch}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": f"Review this diff for logic bugs:\n\n{diff_text}"}],
    )
    if response.stop_reason == "max_tokens":
        print(f"  {fd.filename}: response truncated; findings may be incomplete.")
    text_block = next((b for b in response.content if b.type == "text"), None)
    findings = json.loads(text_block.text).get("findings", []) if text_block else []
    return FileResult(
        findings=findings,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def validate_comments(findings: list[dict], files: list[FileDiff]) -> tuple[list[dict], list[dict]]:
    """Split findings into valid inline comments and fallback (bad line) ones."""
    valid_line_map = {fd.filename: fd.valid_lines for fd in files}
    inline: list[dict] = []
    fallback: list[dict] = []

    for f in findings:
        path = f.get("path", "")
        line = f.get("line")
        body = f.get("body", "")
        if not path or not line or not body:
            continue
        if path in valid_line_map and line in valid_line_map[path]:
            inline.append({"path": path, "line": line, "side": "RIGHT", "body": body})
        else:
            # Line not in diff — demote to PR-level fallback comment
            fallback.append(f)

    return inline, fallback


def usage_footer(total_input: int, total_output: int) -> str:
    input_cost = total_input * 3.00 / 1_000_000
    output_cost = total_output * 15.00 / 1_000_000
    return (
        f"\n\n---\n"
        f"<details><summary>📊 Token usage</summary>\n\n"
        f"| | Tokens | Est. cost |\n"
        f"|---|---|---|\n"
        f"| Input | {total_input:,} | ${input_cost:.4f} |\n"
        f"| Output | {total_output:,} | ${output_cost:.4f} |\n"
        f"| **Total** | **{total_input + total_output:,}** | **${input_cost + output_cost:.4f}** |\n"
        f"\nModel: `{MODEL}` &nbsp;·&nbsp; Region: `us-west-2`"
        f"\n</details>"
    )


def post_review(pull, commit_sha: str, inline: list[dict], fallback: list[dict],
                total_input: int = 0, total_output: int = 0) -> None:
    """Post inline comments; fall back to PR-level comment for rejected ones."""
    summary_lines = [BOT_MARKER]
    total = len(inline) + len(fallback)
    summary_lines.append(f"AI review found {total} potential logic bug(s).")
    summary_lines.append(usage_footer(total_input, total_output))

    if fallback:
        summary_lines.append("\n**The following findings could not be anchored to a diff line:**")
        for f in fallback:
            summary_lines.append(f"\n**`{f['path']}`** (line {f['line']}):\n{f['body']}")

    body = "\n".join(summary_lines)

    if inline:
        try:
            pull.create_review(
                body=body,
                event="COMMENT",
                comments=inline,
            )
            print(f"Posted {len(inline)} inline comment(s), {len(fallback)} fallback(s).")
            return
        except GithubException as exc:
            # 422 — some comment still has a bad line; retry one by one.
            if exc.status != 422:
                raise
            print(f"Bulk review rejected (422); retrying comments one by one.")
            posted = 0
            retry_fallback = list(fallback)
            for comment in inline:
                try:
                    pull.create_review(
                        body=BOT_MARKER,
                        event="COMMENT",
                        comments=[comment],
                    )
                    posted += 1
                except GithubException:
                    retry_fallback.append({
                        "path": comment["path"],
                        "line": comment["line"],
                        "body": comment["body"],
                    })

            # Post any remaining fallbacks as a PR-level comment
            if retry_fallback:
                fb_lines = [BOT_MARKER, f"AI review: {len(retry_fallback)} finding(s) could not be posted inline:\n"]
                for f in retry_fallback:
                    fb_lines.append(f"**`{f['path']}`** (line {f['line']}):\n{f['body']}\n")
                pull.create_issue_comment("\n".join(fb_lines))

            print(f"Posted {posted} inline comment(s), {len(retry_fallback)} fallback(s).")
    else:
        # No inline comments at all — post everything as a PR comment
        pull.create_issue_comment(body)
        print(f"Posted {len(fallback)} fallback comment(s) (no valid inline lines).")


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set")
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        sys.exit("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for Bedrock")

    repo_name = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not repo_name or not event_path:
        sys.exit("GITHUB_REPOSITORY and GITHUB_EVENT_PATH must be set (run inside Actions)")

    with open(event_path, encoding="utf-8") as fh:
        event = json.load(fh)

    pr_data = event.get("pull_request", {})
    pr_number = pr_data.get("number") or event.get("number")
    commit_sha = pr_data.get("head", {}).get("sha", "")
    if not pr_number:
        sys.exit("Could not determine the pull request number from the event payload")

    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo(repo_name)
    pull = repo.get_pull(int(pr_number))

    # Deduplication: skip if already reviewed this commit.
    if commit_sha and already_reviewed(pull, commit_sha):
        print(f"Already reviewed commit {commit_sha[:7]}; skipping.")
        return 0

    files = collect_files(pull)
    if not files:
        print("No reviewable files in this PR; nothing to do.")
        return 0

    print(f"Reviewing {len(files)} file(s)...")
    client = anthropic.AnthropicBedrock()
    all_findings: list[dict] = []
    total_input = total_output = 0
    for fd in files:
        print(f"  {fd.filename}")
        result = find_bugs_for_file(client, fd)
        all_findings.extend(result.findings)
        total_input += result.input_tokens
        total_output += result.output_tokens

    print(f"Tokens used — input: {total_input:,}, output: {total_output:,}")

    if not all_findings:
        print("No logic bugs found.")
        return 0

    inline, fallback = validate_comments(all_findings, files)
    post_review(pull, commit_sha, inline, fallback, total_input, total_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
