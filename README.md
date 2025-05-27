# Sandbox HostAgent

A sandbox daemon for managing Docker containers and provide interaction for Reinforcement Learning agents.

## Features

- Start and shutdown Sandbox
- Execute commands in containers, interactive from a tty
- Retrieve command outputs

## TODO
- Test non-interactive command run (exec_run)

## Get Started
Build the hostagent
```bash
go build
```

Run the hostagent
```bash
./multiturn-rl-hostagent
```

## Example Test Usage
```bash
python3 test_hostagent.py start
python3 test_hostagent.py run "ls -al"
python3 test_hostagent.py output "cmd-001"
python3 test_hostagent.py shutdown
```

Nginx example
```bash
python3 test_hostagent.py
```