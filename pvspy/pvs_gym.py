import asyncio
import json
import re
import copy
from typing import Any, Optional
from pydantic import BaseModel
import websockets

# ----------------------- JSON-RPC DATA STRUCTURES -----------------------

class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: list[Any]
    id: str


class JsonRpcResponse(BaseModel):
    jsonrpc: str
    id: str
    result: Optional[Any] = None
    error: Optional[Any] = None
    method: Optional[Any] = None
    message: Optional[Any] = None

# ----------------------- JSON-RPC CLIENT -----------------------

class JsonRpcClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.msg_id = 1
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def __aenter__(self):
        try:
            self.websocket = await websockets.connect(self.uri,open_timeout=1)
            return self
        except websockets.exceptions.InvalidHandshake as e:
            print(f"Handshake failed: {e}")
            raise
        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")
            raise

    async def __aexit__(self, exc_type, exc, tb):
        if self.websocket:
            await self.websocket.close()

    async def call(self, method: str, params: Any, debug: bool = False) -> JsonRpcResponse:
        req = JsonRpcRequest(method=method, params=params, id=str(self.msg_id))
        await self.websocket.send(req.model_dump_json())
        if debug:
            print(f"\n> {method} {req.params}")

        # if method == "typecheck":
        #     parse_msg, typec_msg = [], []
        #     file_name = params[0]
        #     with open(file_name) as f:
        #         topfname = f.readline().split(":")[0]            
        #     typecheck_result = f"{topfname} typechecked in"
        #     while True:
        #         try:
        #             response = await asyncio.wait_for(self.websocket.recv(), timeout=1)
        #             resp_obj = JsonRpcResponse.model_validate_json(response)
        #             # print(response)
        #             # print(resp_obj)
        #             if resp_obj.error:
        #                 raise RuntimeError("Error in typechecking")
        #             if resp_obj.method and resp_obj.message:
        #                 parse_msg.append(resp_obj.message)
        #                 # if typecheck_result in resp_obj.message:
        #                 #     return '\n'.join(parse_msg)
        #                 return '\n'.join(parse_msg)
        #         except Exception as e:
        #             print('\n'.join(parse_msg))
        #             raise RuntimeError( f"[{type(e).__name__}] Error in typechecking")
        # else:
        response = await self.websocket.recv()

        resp_obj = JsonRpcResponse.model_validate_json(response)
        if debug:
            if resp_obj.error:
                print(f"< [ERROR] {method}: {resp_obj.error}")
            else:
                print(f"< [RESULT] {method}: {resp_obj.result}")
        self.msg_id += 1
        return resp_obj


# ----------------------- PARSING UTILITIES -----------------------
SEQUENT_RE = re.compile(r'^[\[\{](-?\d+)[\]\}]\s*(.*)$')
DELIM = "|-------"
QED_LINE = "This completes the proof of"

def parse_section(section: str):
    lines = [L.rstrip() for L in section.splitlines() if L.strip()]
    items = []
    for ln in lines:
        m = SEQUENT_RE.match(ln)
        if m:
            items.append(ln)
        else:
            # Continuation of the previous line (if any)
            if items:
                items[-1] = (items[-1] + ' ' + ln.strip()).strip()
            else:
                # No prior item; ignore or start a generic bucket
                # Here we ignore stray non-sequent lines
                continue
    return items


def parse_proof_state(commentaries):
    interim = []
    consequents, antecedents = [] , []
    for line in commentaries:
        line = line.strip()
        ## some branch is proved ... remove their interims
        if line.startswith(QED_LINE):
            interim = ["QED"]
        elif DELIM in line:
            commentary = line.split("\n\n")
            th_name, sequents = commentary[0], commentary[1]
            ante_text, cons_text = sequents.split(DELIM)
            antecedents = parse_section(ante_text)
            consequents = parse_section(cons_text)
            interim.extend(antecedents + consequents)
    return interim


class PVSExecutor:
    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.client = None
        self.proof_id = None
        self.formula_name = None
        self.state = None

    async def __aenter__(self):
        self.client = JsonRpcClient(self.uri)
        await self.client.__aenter__()  # sets up the JSON-RPC WebSocket connection
        return self  # now you can use `as pvs` in the async with block

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def typecheck_and_prove_tccs(self, file_name: str):
        async with JsonRpcClient(self.uri) as client:
            await client.call("clear-workspace", ["t"])
            info = await client.call("typecheck", [file_name])
            print(info)
            return info

    async def reset(self, file_name: str, formula_name: str):
        self.formula_name = formula_name
        await self.typecheck_and_prove_tccs(file_name)
        result_obj = await self.client.call("prove-formula", [formula_name])
        if result_obj.result is None:
            raise RuntimeError("Proof-id not detected, check typecheck step")
        self.proof_id = result_obj.result[0]['id']
        print("proving id: ", self.proof_id)
        all_sequents = []
        commentary_info = []
        for step_object in result_obj.result:
            if isinstance(step_object, dict) and 'commentary' in step_object:
                commentary = step_object['commentary']
                sequents = parse_proof_state(commentary)
                all_sequents.extend(sequents)
                commentary_info.extend(commentary)
        return all_sequents, commentary_info


    async def check_proof_status(self) -> bool:
        resp_obj = await self.client.call("proof-status", [self.formula_name])
        return resp_obj.result == "proved"

    #TODO: can we extract state change signals from PVS xx, ?, ....
    ## send as a lisp command to JSON RPC
    # (lisp (status-flag *ps*))
    def get_reward(self, next_state):
        if next_state == ['QED']: ## all branches proved
            return 1.0
        elif 'QED' in next_state: ## major progress , some branches proved
            return 0.5
        elif next_state == self.state: ## bad command no change ( assert )
            return -0.5
        else: ## some progress 
            return 0.2

    async def step(self, command: str):
        """
        Send a tactic step and return:
        - sequents: parsed proof state (next state)
        - is_proved: True if QED reached
        - commentary_info: raw commentary lines
        """
        commentary_info = []
        all_sequents = []

        step_resp_obj = await self.client.call("proof-command", [self.proof_id, command])
        if step_resp_obj.error:
            next_state = copy.deepcopy(self.state)
            reward = -0.5
            truncated = True
            return next_state, reward, False, truncated, commentary_info

        if isinstance(step_resp_obj.result, list):
            for step_object in step_resp_obj.result:
                if isinstance(step_object, dict) and 'commentary' in step_object:
                    commentary = step_object['commentary']
                    sequents = parse_proof_state(commentary)
                    all_sequents.extend(sequents)
                    commentary_info.append(commentary)

        terminated = await self.check_proof_status()
        truncated = False
        if terminated: all_sequents = ["QED"]
        reward = self.get_reward(all_sequents)
        self.state = [s for s in all_sequents if s != "QED"]
        return all_sequents, reward, terminated, truncated, commentary_info