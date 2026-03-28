#!/bin/bash
# Fix all shell scripts and Python files to remove Windows line endings
find /root/vidgen -type f \( -name "*.sh" -o -name "*.py" \) -exec sed -i 's/\r$//' {} \;
echo "Fixed line endings for all .sh and .py files"
