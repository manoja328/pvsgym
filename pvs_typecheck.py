import asyncio
import time, os, shutil
from pvspy.pvs_gym import PVSExecutor
from pvspy.pvs_utils import start_pvs_server, kill_pvs_server 

async def run_typecheck_proof(PORT, file_path):
    async with PVSExecutor("localhost", str(PORT)) as env:
        result = await env.typecheck_and_prove_tccs(file_path)
        return result

def simple_tc(PORT, file_path):
    full_path = os.path.abspath(file_path)
    assert os.path.isfile(full_path), f"File does not exist: {full_path}"
    ## remove bin folder
    file_dir = os.path.dirname(full_path)
    pvsbin_dir = os.path.join(file_dir, "pvsbin")
    if os.path.isdir(pvsbin_dir):   shutil.rmtree(pvsbin_dir)
    print(f"Typechecking {full_path}")
    process = None
    try:
        process = start_pvs_server(PORT)
        time.sleep(1)
        result = asyncio.run(run_typecheck_proof(PORT, full_path))
        print(result)
        print("✅ type check passed.")
    except Exception as e:
        print(f"❌ type check failed: {e}")
    finally:
        if process: kill_pvs_server(process)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python main.py <port> <progress_file>")
        sys.exit(1)

    port = int(sys.argv[1])
    file_path = sys.argv[2]
    simple_tc(port, file_path)

