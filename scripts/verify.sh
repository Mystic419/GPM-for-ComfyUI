#!/usr/bin/env bash
set -e

echo "Running project verification..."

if [ -d "tests" ]; then
  echo "Tests folder found. Add your real test/lint/build commands to scripts/verify.sh"
else
  echo "No tests folder found."
fi

echo "Verification script completed."
