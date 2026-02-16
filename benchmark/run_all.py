#!/usr/bin/env python3
"""Orchestrator: runs all benchmark suites and produces final report.

Suites 1+2 always run (no API key needed).
Suites 3+4 only run if ANTHROPIC_API_KEY is set.
"""

import sys
import time

from conftest import has_api_key, get_know_version


def main():
    version = get_know_version()
    print(f"know-cli Benchmark Suite — v{version}")
    print(f"API key: {'set' if has_api_key() else 'not set (suites 3+4 will be skipped)'}")
    print()

    t0 = time.monotonic()

    # Suite 1: Token Efficiency
    print("─" * 60)
    from bench_token_efficiency import run_suite as run_suite_1
    run_suite_1()

    # Suite 2: Session Dedup
    print()
    print("─" * 60)
    from bench_session_dedup import run_suite as run_suite_2
    run_suite_2()

    # Suites 3+4: Agent E2E + Quality (requires API key)
    if has_api_key():
        print()
        print("─" * 60)
        from bench_agent_e2e import run_suite as run_suite_3
        try:
            run_suite_3()
        except Exception as e:
            print(f"  Suite 3+4 failed: {e}")
    else:
        print()
        print("─" * 60)
        print("Suites 3+4: Skipped (set ANTHROPIC_API_KEY to enable)")

    # Generate report
    print()
    print("─" * 60)
    from bench_report import main as report_main
    report_main()

    elapsed = time.monotonic() - t0
    print(f"\nTotal benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
