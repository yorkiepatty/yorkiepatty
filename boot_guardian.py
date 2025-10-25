"""
boot_guardian.py
----------------
This module runs at the very start of Derek's boot sequence.
It validates all JSON memory files against their schemas before
any other subsystem initializes.
"""

import json
from pathlib import Path
from jsonschema import validate, ValidationError


class BootGuardian:
    def __init__(self, memory_dir="memory", schema_dir="schemas"):
        self.memory_dir = Path(memory_dir)
        self.schema_dir = Path(schema_dir)

    def validate_json_file(self, json_path: Path):
        schema_path = self.schema_dir / f"{json_path.stem}_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"⚠ No schema found for {json_path.name}")

        with open(json_path, "r") as jf, open(schema_path, "r") as sf:
            data = json.load(jf)
            schema = json.load(sf)
            try:
                validate(instance=data, schema=schema)
            except ValidationError as e:
                raise ValueError(
                    f"❌ Validation failed for {json_path.name}\n→ {e.message}"
                )

    def run_full_validation(self):
        json_files = list(self.memory_dir.rglob("*.json"))
        if not json_files:
            print(f"⚠ No JSON memory files found in {self.memory_dir}")
            return

        print(f"🔍 Running integrity check on {len(json_files)} memory JSONs...")

        for json_file in json_files:
            self.validate_json_file(json_file)

        print("✅ All JSON memory files passed validation.")


# You can call this module directly as a script for manual runs:
if __name__ == "__main__":
    guardian = BootGuardian()
    guardian.run_full_validation()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
