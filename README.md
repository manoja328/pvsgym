# PVSgym

https://www.manojacharya.com/pvsgym/

## Build and Install PVS and check if it can run in raw mode:
`./pvs -raw -port 8080`

## Install the pvspy library:

```
cd pvspy
pip install -e .
```

## Typechecking on a file: 
`python pvspy/pvspy/pvs_typecheck.py 8780 agent_expt/imo_1977_p6/trial_2.pvs`

## PVSgym on a specification file with a list of proofs: 

`python pvs_chat_file.py`

### Example run output:
```
(base) ➜  pvs_scripts git:(main) ✗ python ./pvs_chat_file.py
🔁 Starting proof: inequality_lemma (4 steps)
🚀 Starting PVS server...
✅ PVS server started on port 8080 (PID 92351)
jsonrpc='2.0' id='2' result=None error=None method='pvsMessage' message='Restored theory from /Users/e33778/PVS/nasalib/reals/pvsbin/sqrt_exists.bin in 0.08s (load part took 0.00s)'
proving id:  inequality_lemma-2
Goal: ['{1}   FORALL (a, b, c: real): lhs(a, b, c) <= rhs(a, b, c)']
step:0 skeep  (reward = 0.20)
['{1}   lhs(a, b, c) <= rhs(a, b, c)']

step:1 expand "lhs"  (reward = 0.20)
['{1}   -1 * (a ^ 2 * a * c) - b ^ 2 * a * b - c ^ 2 * b * c + a ^ 2 * a * b + b ^ 2 * b * c + c ^ 2 * a * c <= rhs(a, b, c)']

step:2 expand "rhs"  (reward = 0.20)
['{1}   -1 * (a ^ 2 * a * c) - b ^ 2 * a * b - c ^ 2 * b * c + a ^ 2 * a * b + b ^ 2 * b * c + c ^ 2 * a * c <= (9 * sqrt(2) / 32) * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2']

step:3 metit  (reward = 1.00)
['QED']

Run time  = 0.56 secs.
Real time = 1.404 secs.
🛑 Killed process group for PID 92351
```

## Cite our work: 
```
@inproceedings{acharya2025pvsgym,
  title={PVSGym: A Proof Learning Environment},
  author={Acharya, Manoj and Nukala, Karthik and Shankar, Natarajan},
  booktitle={The 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025}
}
```




