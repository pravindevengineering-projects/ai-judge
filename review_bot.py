"""AI PR reviewer.

Reads a pull request's diff via PyGithub, asks Claude (AWS Bedrock) to find
logic and security bugs with one-click GitHub ``suggestion`` fixes, verifies
the findings in a second pass, and posts them as inline review comments.

Per-repo behaviour is configured with an optional ``.ai-judge.yml`` and the
repo's own CLAUDE.md / CONTRIBUTING.md, both read from the PR's base branch so
a PR cannot weaken its own review.

Designed to run inside GitHub Actions on a ``pull_request`` event, where
GITHUB_REPOSITORY and GITHUB_EVENT_PATH are set automatically.
"""

from __future__ import annotations

import csv
import datetime
import fnmatch
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic
import yaml
from github import Auth, Github
from github.GithubException import GithubException

MODEL = os.environ.get("AI_JUDGE_MODEL", "us.anthropic.claude-sonnet-4-6")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

# USD per million tokens. Cache writes cost 1.25x input, cache reads 0.1x.
INPUT_PRICE = float(os.environ.get("AI_JUDGE_INPUT_PRICE_PER_MTOK", "3.00"))
OUTPUT_PRICE = float(os.environ.get("AI_JUDGE_OUTPUT_PRICE_PER_MTOK", "15.00"))

MAX_CHUNK_CHARS = 12_000      # numbered diff sent per file
MAX_CONTEXT_CHARS = 40_000    # full-file context; larger files get diff only
MAX_GUIDANCE_CHARS = 6_000    # repo CLAUDE.md / CONTRIBUTING.md excerpt
MAX_WORKERS = 5
MAX_FINDINGS_PER_FILE = 5

CONFIG_FILE = ".ai-judge.yml"
GUIDANCE_FILES = ("CLAUDE.md", "CONTRIBUTING.md", ".github/CONTRIBUTING.md")
BOT_MARKER = "<!-- ai-judge-review -->"

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
SEVERITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}

_MINIFIED_RE = re.compile(r"\.min\.(js|css)$", re.IGNORECASE)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_TEST_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(_test|\.test|\.spec|_spec)\.[^/]+$|(^|/)test_[^/]+$"
)

# Extensions with no reviewable logic. Repos can re-enable any of these via
# `include_extensions` in .ai-judge.yml.
_SKIP_EXTENSIONS = {
    # Markup / docs
    ".html", ".htm", ".md", ".rst", ".txt", ".rtf", ".apib",
    # Style
    ".css", ".scss", ".sass",
    # Data / config
    ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".lock",
    # Test specs / templates
    ".feature", ".mustache", ".hbs",
    # Parser grammars
    ".g4", ".pegjs",
    # Binary / assets
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".pdf", ".zip", ".tar", ".gz", ".map",
}

_SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "Gemfile.lock",
    "go.sum", "Pipfile.lock", "composer.lock", "gradlew", "gradlew.bat",
}

_C_HINT = "buffer overflows, null pointer dereference, integer overflow/underflow, use-after-free, missing bounds checks, signed/unsigned comparison."
_CPP_HINT = "same as C plus object slicing, virtual dispatch issues, RAII violations, iterator invalidation, dangling references."
_JS_HINT = "`==` vs `===`, undefined/null not handled, missing `await`, async errors silently ignored, off-by-one in array loops, `this` binding issues."
_TS_HINT = "null/undefined not handled, missing `await` on async calls, type assertions (`as`) bypassing runtime checks, off-by-one in array loops."
_REACT_HINT = " In React components also check: stale closures in hooks, missing/incorrect useEffect dependencies, state mutated directly, keys missing or using array index, dangerouslySetInnerHTML with untrusted data."
_SH_HINT = "unquoted variables causing word splitting, missing exit code checks, `pipefail` not set, `$?` read after the wrong command, command injection via eval or unquoted input."

LANGUAGE_HINTS: dict[str, str] = {
    ".py":    "mutable default arguments, `is` vs `==`, silent exception swallowing (`except: pass`), integer division (`/` vs `//`), generator exhaustion, f-string SQL.",
    ".java":  "NullPointerException risks, `equals()` vs `==` for objects, unchecked exceptions swallowed silently, integer division truncation, off-by-one in collections.",
    ".scala": "Option.get on None, non-exhaustive pattern matches, integer division truncation, `equals()` vs `eq()`, mutable shared state.",
    ".kt":    "null safety operator misuse (`!!`), smart cast invalidation, integer division truncation, coroutine cancellation not handled.",
    ".groovy":"NullPointerException, Groovy truth pitfalls (0 and '' are falsy), GString injection, dynamic dispatch surprises.",
    ".go":    "ignored error returns, goroutine leaks, nil pointer dereference, slice off-by-one, `defer` inside loops executing too late, loop variable capture.",
    ".rs":    "`unwrap()`/`expect()` on None/Err without handling, integer overflow in debug vs release, off-by-one in index operations, borrow checker workarounds that hide bugs.",
    ".c":     _C_HINT,
    ".h":     _C_HINT,
    ".cpp":   _CPP_HINT,
    ".cc":    _CPP_HINT,
    ".hpp":   _CPP_HINT,
    ".cs":    "null reference risks, `async void`, missing `await`, `IDisposable` not disposed, LINQ deferred execution surprises, integer division truncation.",
    ".swift": "force unwrap (`!`) on nil, retain cycles from strong self in closures, off-by-one in ranges (`...` vs `..<`), main-thread UI updates.",
    ".php":   "`==` loose comparison pitfalls, unescaped output (XSS), SQL built by string concatenation, null handling, include/require of user-controlled paths.",
    ".ts":    _TS_HINT,
    ".tsx":   _TS_HINT + _REACT_HINT,
    ".js":    _JS_HINT,
    ".mjs":   _JS_HINT,
    ".cjs":   _JS_HINT,
    ".jsx":   _JS_HINT + _REACT_HINT,
    ".rb":    "nil handling, method_missing pitfalls, mutable object sharing, `rescue Exception` catching too broadly.",
    ".pl":    "undef handling, string vs number context confusion, regex capture group off-by-one, wantarray context bugs.",
    ".lua":   "1-based indexing errors, nil handling, table mutation during iteration, local vs global variable shadowing.",
    ".sh":    _SH_HINT,
    ".bash":  _SH_HINT,
    ".ps1":   "unchecked `$?`, null/empty string handling, pipeline errors silently ignored, implicit type coercion.",
    ".bat":   "missing `ERRORLEVEL` checks, variable expansion issues, logic errors in IF/FOR constructs.",
    ".sql":   "UPDATE/DELETE without WHERE clause, NULL comparison using `=` instead of `IS NULL`, missing transaction around multi-statement writes, aggregates without GROUP BY, off-by-one in LIMIT/OFFSET.",
    ".hcl":   "`count` vs `for_each` logic errors, incorrect resource references, missing `depends_on` causing race conditions, incorrect conditional expressions.",
    ".tf":    "`count` vs `for_each` logic errors, incorrect resource references, missing `depends_on`, overly permissive IAM/security-group rules, secrets in plain text.",
    ".yar":   "condition logic inversions, `any of` vs `all of` semantics, overlapping string definitions, incorrect offset calculations.",
    ".yara":  "condition logic inversions, `any of` vs `all of` semantics, overlapping string definitions, incorrect offset calculations.",
    ".xsl":   "XPath predicate logic errors, missing default templates causing silent data loss, incorrect `select` expressions.",
    ".xslt":  "XPath predicate logic errors, missing default templates causing silent data loss, incorrect `select` expressions.",
}

FILENAME_HINTS: dict[str, str] = {
    "Dockerfile": "running as root, `latest` or unpinned base images, secrets copied into layers or passed via ARG/ENV, `apt-get update` in a separate layer from install, missing `--no-cache`/cleanup, wrong COPY order breaking the build cache.",
    "Makefile":   "missing `.PHONY`, recipes not failing on errors in pipelines, unquoted variables, targets with wrong prerequisites, spaces instead of tabs.",
}

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def load_prompt(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8") as fh:
        return fh.read().strip()


REVIEW_PROMPT = load_prompt("review.md")
SECURITY_PROMPT = load_prompt("security.md")
VERIFY_PROMPT = load_prompt("verify.md")
LOGIC_ONLY_PROMPT = 'Report only category "logic" findings for this repository.'

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path exactly as given."},
                    "start_line": {"type": "integer", "description": "First new-file line the fix replaces."},
                    "line": {"type": "integer", "description": "Last new-file line the fix replaces."},
                    "category": {"type": "string", "enum": ["logic", "security"]},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "body": {"type": "string", "description": "Explanation plus a ```suggestion block."},
                },
                "required": ["path", "start_line", "line", "category", "severity", "confidence", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "keep", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class Config:
    ignore_paths: list[str] = field(default_factory=list)
    include_extensions: list[str] = field(default_factory=list)
    min_severity: str = "medium"
    min_confidence: str = "medium"
    max_files: int = 20
    verify: bool = True
    security: bool = True
    extra_hints: dict[str, str] = field(default_factory=dict)
    instructions: str = ""


def load_config(repo, ref: str) -> Config:
    """Read .ai-judge.yml from the repo; fall back to defaults on any problem."""
    cfg = Config()
    raw = fetch_text(repo, CONFIG_FILE, ref)
    if raw is None:
        return cfg
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        print(f"Ignoring invalid {CONFIG_FILE}: {exc}")
        return cfg
    if not isinstance(data, dict):
        print(f"Ignoring {CONFIG_FILE}: top level must be a mapping")
        return cfg

    for key, value in data.items():
        if key in ("ignore_paths", "include_extensions") and isinstance(value, list):
            items = [str(v) for v in value]
            if key == "include_extensions":
                items = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in items]
            setattr(cfg, key, items)
        elif key == "min_severity" and value in SEVERITY_RANK:
            cfg.min_severity = value
        elif key == "min_confidence" and value in CONFIDENCE_RANK:
            cfg.min_confidence = value
        elif key == "max_files" and isinstance(value, int) and value > 0:
            cfg.max_files = value
        elif key in ("verify", "security") and isinstance(value, bool):
            setattr(cfg, key, value)
        elif key == "extra_hints" and isinstance(value, dict):
            cfg.extra_hints = {str(k): str(v) for k, v in value.items()}
        elif key == "instructions" and isinstance(value, str):
            cfg.instructions = value.strip()
        else:
            print(f"Ignoring unknown or invalid {CONFIG_FILE} key: {key}")
    print(f"Loaded {CONFIG_FILE}: {cfg}")
    return cfg


def fetch_text(repo, path: str, ref: str) -> str | None:
    """Return a file's text at ref, or None if missing, binary, or too large."""
    try:
        content = repo.get_contents(path, ref=ref)
    except GithubException:
        return None
    if isinstance(content, list):
        return None
    try:
        return content.decoded_content.decode("utf-8")
    except (AssertionError, UnicodeDecodeError, GithubException):
        return None


def load_guidance(repo, ref: str) -> str:
    parts = []
    for path in GUIDANCE_FILES:
        text = fetch_text(repo, path, ref)
        if text:
            parts.append(f"--- {path} ---\n{text.strip()}")
    return "\n\n".join(parts)[:MAX_GUIDANCE_CHARS]


# ── Diff handling ─────────────────────────────────────────────────────────

@dataclass
class FileDiff:
    filename: str
    hunks: list[str]              # each hunk with new-file line numbers prefixed
    added_lines: set[int]         # lines a comment may anchor to
    line_hunk: dict[int, int]     # RIGHT-side line -> hunk index (for ranges)
    changes: int = 0


def is_reviewable(filename: str, cfg: Config) -> bool:
    if any(fnmatch.fnmatch(filename, pattern) for pattern in cfg.ignore_paths):
        return False
    basename = filename.rsplit("/", 1)[-1]
    ext = os.path.splitext(basename)[1].lower()
    if ext and ext in cfg.include_extensions:
        return True
    if basename in _SKIP_FILENAMES or ext in _SKIP_EXTENSIONS:
        return False
    return not _MINIFIED_RE.search(filename)


def parse_patch(filename: str, patch: str, changes: int = 0) -> FileDiff:
    """Number every diff line with its new-file line number."""
    hunks: list[list[str]] = []
    added: set[int] = set()
    line_hunk: dict[int, int] = {}
    new_line = 0
    for raw in patch.splitlines():
        match = _HUNK_RE.match(raw)
        if match:
            new_line = int(match.group(1))
            hunks.append([raw])
            continue
        if not hunks or raw.startswith("\\"):
            continue
        if raw.startswith("-"):
            hunks[-1].append(f"{'':>6} - {raw[1:]}")
            continue
        marker = "+" if raw.startswith("+") else " "
        if marker == "+":
            added.add(new_line)
        line_hunk[new_line] = len(hunks) - 1
        hunks[-1].append(f"{new_line:>6} {marker} {raw[1:]}")
        new_line += 1
    return FileDiff(filename, ["\n".join(h) for h in hunks], added, line_hunk, changes)


def collect_files(pull, cfg: Config) -> dict[str, FileDiff]:
    files: dict[str, FileDiff] = {}
    for f in pull.get_files():
        if f.status == "removed" or not f.patch or not is_reviewable(f.filename, cfg):
            continue
        files[f.filename] = parse_patch(f.filename, f.patch, f.changes)
    return files


def restrict_to_delta(repo, pr_files: dict[str, FileDiff], prev_sha: str,
                      head_sha: str) -> list[FileDiff] | None:
    """Files changed since the last review, limited to the PR's own changes.

    Returns None when the previous commit is no longer an ancestor of head
    (force-push / rebase), meaning a full review is needed.
    """
    try:
        comparison = repo.compare(prev_sha, head_sha)
    except GithubException:
        return None
    if comparison.status != "ahead":
        return None

    delta: list[FileDiff] = []
    for f in comparison.files:
        pr_fd = pr_files.get(f.filename)
        if pr_fd is None or not f.patch:
            continue
        fd = parse_patch(f.filename, f.patch, pr_fd.changes)
        # Lines brought in by merging the base branch are not in the PR diff.
        fd.added_lines &= pr_fd.added_lines
        if not fd.added_lines:
            continue
        fd.line_hunk = pr_fd.line_hunk
        delta.append(fd)
    return delta


def rank_files(files: list[FileDiff]) -> list[FileDiff]:
    """Source before tests, then most-changed first."""
    return sorted(files, key=lambda fd: (bool(_TEST_RE.search(fd.filename)), -fd.changes))


def truncate_hunks(hunks: list[str], limit: int) -> tuple[str, bool]:
    kept: list[str] = []
    size = 0
    for hunk in hunks:
        if kept and size + len(hunk) > limit:
            return "\n".join(kept), True
        kept.append(hunk[:limit])
        size += len(hunk) + 1
    return "\n".join(kept), False


def number_lines(text: str) -> str:
    return "\n".join(f"{i:>6}  {line}" for i, line in enumerate(text.splitlines(), 1))


def hint_for(filename: str, cfg: Config) -> str:
    basename = filename.rsplit("/", 1)[-1]
    ext = os.path.splitext(basename)[1].lower()
    hints = [FILENAME_HINTS.get(basename) or LANGUAGE_HINTS.get(ext, "")]
    hints.append(cfg.extra_hints.get(ext) or cfg.extra_hints.get(basename, ""))
    return " ".join(h for h in hints if h)


# ── Claude calls ──────────────────────────────────────────────────────────

@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    def add(self, other: Usage) -> None:
        self.input += other.input
        self.output += other.output
        self.cache_write += other.cache_write
        self.cache_read += other.cache_read

    @property
    def total_input(self) -> int:
        return self.input + self.cache_write + self.cache_read

    def cost(self) -> float:
        return (
            self.input * INPUT_PRICE
            + self.cache_write * INPUT_PRICE * 1.25
            + self.cache_read * INPUT_PRICE * 0.10
            + self.output * OUTPUT_PRICE
        ) / 1_000_000


def build_system(cfg: Config, guidance: str) -> list[dict]:
    text = REVIEW_PROMPT.replace("{{SECURITY}}", SECURITY_PROMPT if cfg.security else LOGIC_ONLY_PROMPT)
    if cfg.instructions:
        text += f"\n\nRepository maintainers' review instructions:\n{cfg.instructions}"
    if guidance:
        text += (
            "\n\nThe repository's own contributor guidance is below. Use it to "
            "understand intended behaviour and conventions; still report only "
            f"defects, not style.\n<repository_guidance>\n{guidance}\n</repository_guidance>"
        )
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def call_claude(client, system, user: str, schema: dict, max_tokens: int) -> tuple[dict, Usage]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user}],
    )
    u = response.usage
    usage = Usage(
        input=u.input_tokens,
        output=u.output_tokens,
        cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("response truncated at max_tokens")
    text_block = next((b for b in response.content if b.type == "text"), None)
    return (json.loads(text_block.text) if text_block else {}), usage


@dataclass
class FileResult:
    filename: str
    findings: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    below_threshold: int = 0
    rejected_by_verify: int = 0
    error: str = ""


def review_file(client, repo, head_sha: str, fd: FileDiff, cfg: Config, system) -> FileResult:
    result = FileResult(fd.filename)
    try:
        diff, truncated = truncate_hunks(fd.hunks, MAX_CHUNK_CHARS)
        if truncated:
            print(f"  {fd.filename}: diff truncated at a hunk boundary")

        source = fetch_text(repo, fd.filename, head_sha)
        if source and len(source) <= MAX_CONTEXT_CHARS:
            context = f"<full_file>\n{number_lines(source)}\n</full_file>"
        else:
            context = "(Full file not included: too large or unavailable. Review from the diff only.)"

        hint = hint_for(fd.filename, cfg)
        hint_text = f"Language-specific things to check: {hint}\n\n" if hint else ""
        user = f"File: {fd.filename}\n\n{hint_text}{context}\n\n<diff>\n{diff}\n</diff>"

        data, usage = call_claude(client, system, user, OUTPUT_SCHEMA, 8192)
        result.usage.add(usage)

        findings = []
        for f in data.get("findings", []):
            f["path"] = fd.filename
            if (SEVERITY_RANK.get(f.get("severity"), 0) < SEVERITY_RANK[cfg.min_severity]
                    or CONFIDENCE_RANK.get(f.get("confidence"), 0) < CONFIDENCE_RANK[cfg.min_confidence]):
                result.below_threshold += 1
                continue
            findings.append(f)

        if cfg.verify and findings:
            listed = "\n\n".join(
                f"Finding {i}: lines {f['start_line']}-{f['line']} "
                f"[{f['severity']} {f['category']}]\n{f['body']}"
                for i, f in enumerate(findings)
            )
            verify_user = f"{user}\n\n<findings>\n{listed}\n</findings>"
            verdicts, usage = call_claude(client, VERIFY_PROMPT, verify_user, VERIFY_SCHEMA, 4096)
            result.usage.add(usage)
            keep = {v["index"] for v in verdicts.get("verdicts", []) if v.get("keep")}
            result.rejected_by_verify = len(findings) - len(keep & set(range(len(findings))))
            findings = [f for i, f in enumerate(findings) if i in keep]

        findings.sort(key=lambda f: -SEVERITY_RANK.get(f.get("severity"), 0))
        result.findings = findings[:MAX_FINDINGS_PER_FILE]
    except Exception as exc:  # one bad file must not sink the whole review
        result.error = f"{type(exc).__name__}: {exc}"
        print(f"  {fd.filename}: review failed — {result.error}")
    return result


# ── Posting ───────────────────────────────────────────────────────────────

def format_body(f: dict) -> str:
    severity = f.get("severity", "medium")
    icon = SEVERITY_ICON.get(severity, "")
    return f"**{icon} {severity.title()} · {f.get('category', 'logic').title()}**\n\n{f['body']}"


def build_comments(findings: list[dict], files: list[FileDiff]) -> tuple[list[dict], list[dict]]:
    """Split findings into anchorable inline comments and PR-level fallbacks."""
    by_name = {fd.filename: fd for fd in files}
    inline: list[dict] = []
    fallback: list[dict] = []
    for f in findings:
        fd = by_name.get(f.get("path", ""))
        line = f.get("line")
        start = f.get("start_line") or line
        if fd is None or line not in fd.added_lines or start > line:
            fallback.append(f)
            continue
        comment = {"path": fd.filename, "line": line, "side": "RIGHT", "body": format_body(f)}
        if start < line:
            if start not in fd.line_hunk or fd.line_hunk[start] != fd.line_hunk.get(line):
                fallback.append(f)  # a range suggestion anchored to one line would corrupt code
                continue
            comment.update(start_line=start, start_side="RIGHT")
        inline.append(comment)
    return inline, fallback


def usage_footer(usage: Usage) -> str:
    return (
        f"\n\n<details><summary>📊 Token usage</summary>\n\n"
        f"| | Tokens |\n|---|---|\n"
        f"| Input | {usage.input:,} |\n"
        f"| Cache write / read | {usage.cache_write:,} / {usage.cache_read:,} |\n"
        f"| Output | {usage.output:,} |\n"
        f"| **Est. cost** | **${usage.cost():.4f}** |\n"
        f"\nModel: `{MODEL}` &nbsp;·&nbsp; Region: `{REGION}`\n</details>"
    )


def fallback_section(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["\n\n**Findings that could not be anchored to a diff line:**"]
    for f in items:
        body = f.get("body", "")
        if "severity" in f:
            body = format_body(f)
        lines.append(f"\n**`{f['path']}`** (line {f['line']}):\n{body}")
    return "\n".join(lines)


def post_review(pull, commit, summary: str, inline: list[dict], fallback: list[dict]) -> None:
    """Always posts a review so dedup and delta review have an anchor commit."""
    kwargs = {"commit": commit, "event": "COMMENT"}
    try:
        pull.create_review(body=summary + fallback_section(fallback),
                           comments=inline, **kwargs)
        print(f"Posted review: {len(inline)} inline, {len(fallback)} fallback.")
        return
    except GithubException as exc:
        if exc.status != 422 or not inline:
            raise
    print("Bulk review rejected (422); retrying comments one by one.")
    rejected = []
    for comment in inline:
        try:
            pull.create_review(body=BOT_MARKER, comments=[comment], **kwargs)
        except GithubException:
            rejected.append(comment)
    pull.create_review(body=summary + fallback_section(fallback + rejected), **kwargs)
    print(f"Posted review: {len(inline) - len(rejected)} inline, {len(fallback) + len(rejected)} fallback.")


def build_summary(findings: list[dict], results: list[FileResult], mode: str,
                  skipped_by_cap: int, usage: Usage) -> str:
    lines = [BOT_MARKER]
    if findings:
        counts = {}
        for f in findings:
            counts[f.get("severity", "medium")] = counts.get(f.get("severity", "medium"), 0) + 1
        breakdown = ", ".join(
            f"{SEVERITY_ICON[s]} {counts[s]} {s}" for s in SEVERITY_RANK if s in counts
        )
        lines.append(f"**AI Judge** found {len(findings)} issue(s) in {len(results)} file(s) ({mode} review): {breakdown}")
    else:
        lines.append(f"✅ **AI Judge** found no issues in {len(results)} file(s) ({mode} review).")

    notes = []
    dropped = sum(r.below_threshold for r in results)
    rejected = sum(r.rejected_by_verify for r in results)
    if dropped:
        notes.append(f"{dropped} low-severity/low-confidence finding(s) hidden by threshold")
    if rejected:
        notes.append(f"{rejected} finding(s) discarded by the verification pass")
    if skipped_by_cap:
        notes.append(f"⚠️ {skipped_by_cap} file(s) not reviewed (cost cap, smallest/test files skipped first)")
    failed = [r.filename for r in results if r.error]
    if failed:
        notes.append("⚠️ review failed for: " + ", ".join(f"`{n}`" for n in failed))
    if notes:
        lines.append("\n" + "\n".join(f"- {n}" for n in notes))
    lines.append(usage_footer(usage))
    return "\n".join(lines)


# ── Dedup / history ───────────────────────────────────────────────────────

def bot_reviews(pull) -> list:
    return [
        r for r in pull.get_reviews()
        if r.user.login.endswith("[bot]") and BOT_MARKER in (r.body or "")
    ]


# ── Logging ───────────────────────────────────────────────────────────────

LOG_FILE = os.environ.get("AI_JUDGE_LOG_FILE", "review_log.csv")
LOG_FIELDS = ["timestamp", "repo", "pr_number", "files_reviewed", "findings", "input_tokens", "output_tokens", "cost_usd"]


def append_review_log(repo_name: str, pr_number: int, files_reviewed: int,
                      findings: int, usage: Usage) -> None:
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo": repo_name,
            "pr_number": pr_number,
            "files_reviewed": files_reviewed,
            "findings": findings,
            "input_tokens": usage.total_input,
            "output_tokens": usage.output,
            "cost_usd": f"{usage.cost():.4f}",
        })
    print(f"Review log updated: {LOG_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────

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
    if not pr_number:
        sys.exit("Could not determine the pull request number from the event payload")

    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo(repo_name)
    pull = repo.get_pull(int(pr_number))
    commit_sha = pr_data.get("head", {}).get("sha") or pull.head.sha

    if pull.user.login.endswith("[bot]"):
        print(f"PR opened by bot ({pull.user.login}); skipping.")
        return 0

    previous = bot_reviews(pull)
    if any(r.commit_id == commit_sha for r in previous):
        print(f"Already reviewed commit {commit_sha[:7]}; skipping.")
        return 0

    # Config and guidance come from the base branch so a PR can't weaken its own review.
    cfg = load_config(repo, pull.base.ref)
    guidance = load_guidance(repo, pull.base.ref)

    pr_files = collect_files(pull, cfg)
    files: list[FileDiff] | None = None
    mode = "full"
    prev_sha = previous[-1].commit_id if previous else None
    if prev_sha:
        files = restrict_to_delta(repo, pr_files, prev_sha, commit_sha)
        if files is None:
            print(f"Previous review {prev_sha[:7]} is not an ancestor of head; full review.")
        else:
            mode = "incremental"
            print(f"Incremental review {prev_sha[:7]}..{commit_sha[:7]}: {len(files)} file(s).")
    if files is None:
        files = list(pr_files.values())

    if not files:
        print("No reviewable changes; nothing to do.")
        return 0

    files = rank_files(files)
    skipped_by_cap = max(0, len(files) - cfg.max_files)
    files = files[:cfg.max_files]

    print(f"Reviewing {len(files)} file(s) with up to {MAX_WORKERS} in parallel...")
    client = anthropic.AnthropicBedrock()
    system = build_system(cfg, guidance)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(
            lambda fd: review_file(client, repo, commit_sha, fd, cfg, system), files
        ))

    usage = Usage()
    findings: list[dict] = []
    for r in results:
        usage.add(r.usage)
        findings.extend(r.findings)
    print(f"Tokens — input: {usage.total_input:,}, output: {usage.output:,}, est. ${usage.cost():.4f}")

    summary = build_summary(findings, results, mode, skipped_by_cap, usage)
    inline, fallback = build_comments(findings, files)
    post_review(pull, repo.get_commit(commit_sha), summary, inline, fallback)
    append_review_log(repo_name, pr_number, len(files), len(findings), usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
