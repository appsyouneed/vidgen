#!/usr/bin/env python3
"""Run this to change your vidgen password: python3 change-password.py"""
import os, getpass, bcrypt

_CRED_FILE = "/root/vidgen/.credentials"

if os.path.exists(_CRED_FILE):
    current = getpass.getpass("Current password: ").encode()
    with open(_CRED_FILE, "rb") as f:
        stored = f.read().strip()
    if not bcrypt.checkpw(current, stored):
        print("Wrong password.")
        raise SystemExit(1)

while True:
    pw = getpass.getpass("New password: ")
    pw2 = getpass.getpass("Confirm new password: ")
    if pw != pw2:
        print("Passwords do not match.")
        continue
    if len(pw) < 8:
        print("Must be at least 8 characters.")
        continue
    break

hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12))
fd = os.open(_CRED_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "wb") as f:
    f.write(hashed)
print("Password updated. Restart vidgen for it to take effect.")
