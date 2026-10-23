"""python -m helioguard.harness_cli --model <name> --split eval --n 50  (the engine harness, HelioGuard domain)."""
from verispace_engine.harness.eval import main

if __name__ == "__main__":
    main(default_domain="helioguard.harness_env")
