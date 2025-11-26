# Windows Installation Fix Guide

## Problem
You're encountering the error: `ERROR: Compiler cl cannot compile programs` when trying to install NumPy and ml-dtypes. This happens because pip is trying to build packages from source, which requires Microsoft Visual Studio C++ Build Tools.

## Quick Solutions (Try in order)

### Solution 1: Upgrade pip and use pre-built wheels (RECOMMENDED)
This is the fastest and easiest solution:

```bash
# Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Install packages with pre-built wheels
pip install --only-binary :all: numpy==1.26.4
pip install --only-binary :all: ml-dtypes==0.3.2

# Then install the rest of requirements
pip install -r requirements.txt
```

### Solution 2: Install specific package versions with wheels
If Solution 1 doesn't work, try installing packages individually:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install numpy with no build isolation
pip install numpy==1.26.4 --prefer-binary

# Install ml-dtypes
pip install ml-dtypes==0.3.2 --prefer-binary

# Install other problematic packages
pip install PyAudio==0.2.14 --prefer-binary

# Then install remaining requirements
pip install -r requirements.txt
```

### Solution 3: Use Python 3.11 or 3.10
NumPy 1.26.4 has pre-built wheels for Python 3.10 and 3.11 but may have issues with newer Python versions on Windows.

```bash
# Check your Python version
python --version

# If you're using Python 3.12+, consider downgrading to Python 3.11
# Download from: https://www.python.org/downloads/
```

### Solution 4: Install Visual Studio Build Tools
If you must build from source:

1. Download **Microsoft C++ Build Tools**: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select **"Desktop development with C++"**
4. Install and restart your computer
5. Try installing packages again:
   ```bash
   pip install -r requirements.txt
   ```

### Solution 5: Use Anaconda/Miniconda (Alternative approach)
If all else fails, use conda which has pre-compiled binaries:

```bash
# Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# Create a new environment
conda create -n yorkiepatty python=3.11
conda activate yorkiepatty

# Install packages via conda (they're pre-compiled)
conda install numpy scipy scikit-learn pandas
conda install -c conda-forge librosa

# Then install remaining packages with pip
pip install -r requirements.txt
```

## PyAudio Specific Fix
PyAudio often has similar issues on Windows. If it fails:

```bash
# Install PyAudio from unofficial wheels
pip install pipwin
pipwin install pyaudio
```

Or download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

## Verification
After installation, verify:

```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import pyaudio; print('PyAudio OK')"
```

## Common Issues

### Issue: "Failed to build ml-dtypes"
**Fix:** `pip install ml-dtypes --prefer-binary`

### Issue: "error: Microsoft Visual C++ 14.0 or greater is required"
**Fix:** Install Visual Studio Build Tools (Solution 4) OR use pre-built wheels (Solution 1)

### Issue: PyAudio fails to install
**Fix:** Use pipwin or download pre-compiled wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

## Recommended Installation Order

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 2. Install core scientific packages
pip install --prefer-binary numpy==1.26.4
pip install --prefer-binary scipy==1.15.3
pip install --prefer-binary pandas==2.3.3

# 3. Install ML packages
pip install --prefer-binary scikit-learn==1.7.2
pip install --prefer-binary ml-dtypes==0.3.2

# 4. Install audio packages (may need special handling)
pip install pipwin
pipwin install pyaudio

# 5. Install remaining packages
pip install -r requirements.txt
```

## Still Having Issues?

1. **Check Python Version**: Use Python 3.10 or 3.11 for best compatibility
2. **Use Virtual Environment**: Always use a venv to avoid conflicts
3. **Check Internet Connection**: Some packages are large
4. **Clear Cache**: Run `pip cache purge` and try again
5. **Try Conda**: Consider using Anaconda/Miniconda instead of pip

## Environment Setup (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Upgrade pip in venv
python -m pip install --upgrade pip setuptools wheel

# Install packages
pip install -r requirements.txt --prefer-binary
```
