# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This repository is a proof-of-concept that has not been implemented yet. As of this writing it contains only an intent description (`claude.md`) and no source code, dependency manifest, tests, or CI config. Update this file with real commands and architecture once code lands.

## Intended purpose

An automated PR reviewer that posts review comments with one-click code suggestions on GitHub pull requests. Delivered in two forms:

- A **Python script** that can be run directly against a PR.
- A **reusable GitHub Action** that wraps the script so other repos can invoke it in their workflows.

## Intended stack

- **Python** — implementation language.
- **PyGithub** — reads PR diffs/files and writes review comments and suggestions via the GitHub API.
- **Anthropic API** — generates the review feedback and code suggestions. Use the latest Claude models (e.g. `claude-opus-4-8`); consult the `claude-api` skill for current model IDs, params, and SDK usage before writing API calls.

## Notes for implementation

- GitHub's "suggested change" one-click UI requires review comments formatted as ```suggestion fenced blocks anchored to specific diff lines — the reviewer must map Anthropic's output back to file paths and line ranges in the diff.
- A reusable GitHub Action needs an `action.yml` at the repo root defining inputs (e.g. `anthropic-api-key`, `github-token`) and the run entrypoint.

Once source, dependencies, and tests exist, replace the sections above with the actual build/lint/test commands (including how to run a single test) and the concrete architecture.
