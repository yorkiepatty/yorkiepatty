# Runtime Error Fixes

This document describes the fixes applied to resolve runtime errors in the yorkiepatty project.

## Issues Fixed

### 1. Anthropic Client "proxies" TypeError

**Error:**
```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

**Root Cause:**
The anthropic library version 0.39.0 was incompatible with httpx versions 0.27.0 and later. The httpx library removed the deprecated `proxies` parameter in version 0.27.0, but anthropic 0.39.0 was still trying to use it.

**Solution:**
Upgraded the anthropic library from version 0.39.0 to 0.75.0, which includes compatibility fixes for httpx 0.27.0+ (introduced in anthropic 0.45.2).

**Changes:**
- `requirements.txt`: Updated `anthropic==0.39.0` → `anthropic==0.75.0`
- `requirements.txt`: Updated `httpx==0.27.2` → `httpx==0.28.1`

**References:**
- [Anthropic SDK Issue #767](https://github.com/anthropics/anthropic-sdk-python/issues/767)
- [Anthropic SDK Releases](https://github.com/anthropics/anthropic-sdk-python/releases)

### 2. python-dotenv Parsing Error at Line 28

**Error:**
```
python-dotenv could not parse statement starting at line 28
```

**Root Cause:**
The .env file contains a syntax error at line 28. Common causes include:
- Unquoted values with special characters
- Missing quotes for values with spaces
- Incorrect comment formatting
- Special characters like `&`, `|`, `<`, `>`, `$`, `` ` ``, `\`, `"`, `'`, or spaces without quotes

**Solution:**
Check your `.env` file at line 28 and ensure proper formatting. Use `.env.example` as a reference.

**Correct .env Formatting Rules:**
1. Use `KEY=value` format (no spaces around `=`)
2. Quote values with special characters: `KEY="value with special chars"`
3. Do NOT use single quotes for variable expansion
4. Comments must start with `#` at the beginning of the line
5. Avoid special characters without quotes

**Example:**
```env
# Correct
API_KEY="sk-1234567890abcdef"
DATABASE_URL="postgresql://user:pass@localhost/dbname"
FEATURE_FLAG=true

# Incorrect
API_KEY=sk-1234567890abcdef   # Comment here causes error
DATABASE_URL = "postgresql://..."  # Spaces around = cause error
FEATURE_FLAG='true'  # Single quotes can cause issues
```

## Installation Instructions

After pulling these changes, update your Python environment:

```bash
# On Windows (using venv)
cd C:\Users\yorki\sunnyfolder
venv311\Scripts\activate
pip install --upgrade -r requirements.txt

# On Linux/macOS
cd /path/to/yorkiepatty
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

## Verification

To verify the fixes:

1. **Check anthropic version:**
```bash
pip show anthropic | grep Version
# Should show: Version: 0.75.0
```

2. **Check httpx version:**
```bash
pip show httpx | grep Version
# Should show: Version: 0.28.1
```

3. **Test the fixes:**
```python
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
print("✓ Anthropic client initialized successfully!")
```

## Additional Notes

- The `.env.example` file has been created as a reference for proper environment variable formatting
- If you continue to experience .env parsing errors, check for hidden characters or encoding issues (ensure UTF-8)
- Always keep your API keys secure and never commit them to version control

## Version History

- **2025-11-26**: Fixed anthropic/httpx compatibility issue
  - Upgraded anthropic: 0.39.0 → 0.75.0
  - Upgraded httpx: 0.27.2 → 0.28.1
  - Created .env.example template
