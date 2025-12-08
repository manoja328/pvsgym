import os
import sys, time
import subprocess
import websockets
import asyncio
import os, signal

async def is_pvs_alive(uri, timeout=1):
    try:
        async with websockets.connect(uri, ping_timeout=timeout):
            return True
    except Exception:
        raise RuntimeError("❌ PVS server is down or unresponsive.")

def start_pvs_server(port=8080):
    print("🚀 Starting PVS server...")
    pvs_dir = os.path.expanduser("~/PVS")
    cmd = ["./pvs", "-raw", "-port", str(port)]
    process = subprocess.Popen(
        cmd,
        cwd=pvs_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(f"✅ PVS server started on port {port} (PID {process.pid})")
    return process

def kill_pvs_server(proc):
    if proc is None: return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait()
        print(f"🛑 Killed process group for PID {proc.pid}")
    except Exception as e:
        print(f"❌ Failed to kill process group: {e}")

def print_help():
    print("Usage:")
    print("  python pvs_control.py start      # start the PVS server")
    print("  python pvs_control.py stop       # stop the PVS server")
    print("  python pvs_control.py restart    # restart the PVS server")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "start":
        start_pvs_server()
    elif command == "stop":
        kill_pvs_server()
    elif command == "restart":
        kill_pvs_server()
        time.sleep(2)
        start_pvs_server()
    else:
        print(f"❌ Unknown command: {command}")
        print_help()