# PVSgym

## Install the pvspy library:

```
cd pvspy
pip install -e .
```

To start PVS API mode start PVS in raw mode:

`./pvs -raw -port 8080`

## To run PVSgym on a specification file with a list of proofs: 

`python pvs_chat_file.py`

## To run typechecking on a file: 

`python pvspy/pvspy/pvs_typecheck.py 8780 agent_expt/imo_1977_p6/trial_2.pvs`
