#!/usr/bin/env python3
"""
Derek Module Audit Script
Checks which modules are available, which can be imported, and which are being used in main.py
"""

import os
import sys
import importlib
from pathlib import Path

# Get all Python modules in the project
def get_all_python_modules():
    """Find all .py files in the current directory"""
    modules = []
    for file in Path('.').glob('*.py'):
        if file.name != '__pycache__' and not file.name.startswith('.'):
            module_name = file.stem
            if module_name != 'module_audit':  # Exclude this script
                modules.append(module_name)
    return sorted(modules)

def test_module_import(module_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, "✅ OK"
    except ImportError as e:
        return False, f"❌ ImportError: {str(e)[:60]}..."
    except Exception as e:
        return False, f"⚠️  Other error: {str(e)[:60]}..."

def get_imported_in_main():
    """Get modules imported in main.py"""
    imported = []
    try:
        with open('main.py', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('from ') and ' import ' in line:
                    module = line.split('from ')[1].split(' import')[0].strip()
                    if not module.startswith('.') and not module in ['pathlib', 'typing', 'fastapi', 'pydantic', 'boto3', 'datetime', 'tempfile', 'uuid']:
                        imported.append(module)
                elif line.startswith('import ') and not line.startswith('import sys') and not line.startswith('import logging') and not line.startswith('import time') and not line.startswith('import os') and not line.startswith('import boto3') and not line.startswith('import tempfile') and not line.startswith('import uuid'):
                    module = line.split('import ')[1].split('.')[0].strip()
                    imported.append(module)
    except FileNotFoundError:
        pass
    return sorted(list(set(imported)))

def main():
    print("🔍 Derek Module Audit Report")
    print("=" * 80)
    
    # Get all modules
    all_modules = get_all_python_modules()
    imported_modules = get_imported_in_main()
    
    print(f"\n📊 Summary:")
    print(f"   Total Python modules found: {len(all_modules)}")
    print(f"   Modules imported in main.py: {len(imported_modules)}")
    print(f"   Modules NOT being used: {len(set(all_modules) - set(imported_modules))}")
    
    print(f"\n🟢 Modules CURRENTLY IMPORTED in main.py:")
    for module in imported_modules:
        if module in all_modules:
            can_import, status = test_module_import(module)
            print(f"   {module:<30} {status}")
        else:
            print(f"   {module:<30} ❌ File not found")
    
    print(f"\n🔴 Modules AVAILABLE but NOT IMPORTED:")
    unused_modules = sorted(set(all_modules) - set(imported_modules))
    for module in unused_modules:
        can_import, status = test_module_import(module)
        print(f"   {module:<30} {status}")
    
    print(f"\n💡 Suggestions:")
    print(f"   - Review unused modules to see if they should be integrated")
    print(f"   - Remove or archive modules that are no longer needed")
    print(f"   - Fix any import errors in available modules")

if __name__ == "__main__":
    main()