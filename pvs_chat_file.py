import json
import asyncio
import time
from pvspy.pvs_gym import PVSExecutor
from pvspy.pvs_utils import start_pvs_server, kill_pvs_server 

async def run_single_proof(PORT, file_path, formula_name, proof_commands):
    async with PVSExecutor("localhost", str(PORT)) as env:
        rollout = []
        sequents, info = await env.reset(file_path, formula_name)
        print(f"Goal: {sequents}")
        idx = 0
        done = False
        while not done:
            if idx >= len(proof_commands):
                raise IndexError("Command list index out of range")
            command = proof_commands[idx]
            sequents, reward, terminated, truncated, info = await env.step(command)
            print(f"step:{idx} {command[1:-1]}  (reward = {reward:.2f})")
            print(sequents)
            print()
            done = terminated or truncated
            idx += 1
            if done: ## last commentary has time information
                print('\n'.join(info[-1][-2:]))
        return rollout

if __name__ == "__main__":

    file_path = "examples/min2f_imo_2006_p3.pvs"
    formula_name = "inequality_lemma"
    proof_commands = [
                '(skeep)',
                '(expand "lhs")',
                '(expand "rhs")',
                '(metit)',
            ]

    PORT = 8080
    print(f"🔁 Starting proof: {formula_name} ({len(proof_commands)} steps)")
    try:
        process = start_pvs_server(PORT)
        time.sleep(1)
        rollout = asyncio.run(run_single_proof(PORT, file_path, formula_name, proof_commands))
    except Exception as e:
        error_msg = f"❌ Error in formula `{formula_name}`: {e}"
        print(error_msg)
    finally:
        kill_pvs_server(process)
