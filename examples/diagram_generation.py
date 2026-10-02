"""Thin entry point -- see diagram_gen/__init__.py for the architecture
overview. Kept at this path so `python examples/diagram_generation.py`
(the documented invocation) still works unchanged."""

from diagram_gen.cli import main

if __name__ == "__main__":
    main()
