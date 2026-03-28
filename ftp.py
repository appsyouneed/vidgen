import subprocess
import sys
import os

# 1. Automatic Dependency Check & Install
try:
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import FTPServer
except ImportError:
    print("pyftpdlib not found. Installing...")
    subprocess.check_call(["apt", "install", "-y", "python3-pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyftpdlib", "--break-system-packages"])
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import FTPServer

def main():
    # 2. Setup credentials
    authorizer = DummyAuthorizer()
    
    # Replace 'YOUR_PASSWORD' with your desired password
    # "." ensures it shares the current folder and all subfolders
    # perm='elradfmw' provides full read, write, and list permissions
    authorizer.add_user("admin", "YOUR_PASSWORD", ".", perm="elradfmw")

    # 3. Setup handler
    handler = FTPHandler
    handler.authorizer = authorizer
    handler.banner = "FTP Server is ready."

    # 4. Start server
    # 0.0.0.0 makes it public; 2121 avoids needing root privileges for port 21
    address = ("0.0.0.0", 2121)
    server = FTPServer(address, handler)

    print(f"FTP server active at: {os.getcwd()}")
    print("Accessible on port 2121")
    server.serve_forever()

if __name__ == "__main__":
    main()
    