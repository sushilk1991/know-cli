# Tests for know-cli

This directory contains tests that are **NOT** included in the distributed package.

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
cd /path/to/know-cli
python -m pytest tests/ -v

# Run only efficiency tests
python -m pytest tests/test_efficiency.py -v -s

# Run with performance benchmarks
python -m pytest tests/test_efficiency.py::TestIndexingEfficiency -v -s
```

## Test Categories

### `test_efficiency.py`
Performance and caching efficiency tests:
- Initial scan performance
- Cached scan performance  
- Incremental update performance
- Cache correctness verification
- Large fileset handling

These tests measure:
- Time taken for operations
- Cache hit rates
- Speedup factors
- Memory usage patterns

## Notes

- Tests create temporary directories and clean up after
- Some tests create many files (50-200) to measure performance at scale
- Tests verify both speed AND correctness - cached results must match fresh scans
