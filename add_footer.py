#!/usr/bin/env python3
"""
Add copyright footer to all Python modules in The Christman AI Project
"""

import os
import glob

FOOTER = '''
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
'''

def has_footer(content):
    """Check if file already has the footer"""
    return "The Christman AI Project" in content

def add_footer_to_file(filepath):
    """Add footer to a Python file if it doesn't already have it"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_footer(content):
            print(f"⏭️  Skipped (already has footer): {filepath}")
            return False
        
        # Add footer at the end
        new_content = content.rstrip() + '\n' + FOOTER
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Added footer to: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

def main():
    # Get all Python files in the current directory (not subdirectories)
    python_files = glob.glob('*.py')
    
    # Exclude this script itself
    python_files = [f for f in python_files if f != 'add_footer.py']
    
    print(f"Found {len(python_files)} Python files\n")
    
    added = 0
    skipped = 0
    
    for filepath in sorted(python_files):
        if add_footer_to_file(filepath):
            added += 1
        else:
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Added footer to {added} files")
    print(f"⏭️  Skipped {skipped} files (already had footer)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
