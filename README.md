# Ratchet

A harness that rewrites itself, inside limits it cannot widen.

A working agent answers analytics questions over MongoDB data. A second agent,
the editor, reads the failures and proposes one change at a time to the harness
itself: rules, vocabulary, context policy, tool access, budgets. A change is
kept only if it fixes held-out test questions without breaking any that worked,
and it can never exceed the organization's policy.

Built for the MongoDB hackathon, Statement 1 (Recursive Harnessing).

## What is tested and what is not

| Part | Status |
|---|---|
| The loop, the gate, the policy ceiling, memory scoping, rollback (`ratchet/*.py`) | Tested offline, 27 tests |
| Model adapters' JSON handling and tool loop (`ratchet/live/llm.py`) | Tested with a fake model |
| Staging memory writes until the gate accepts (`StagedMemory`) | Tested against local memory |
| Calls to a real model API | **Not run** |
| Atlas adapter (`ratchet/live/atlas.py`) | **Not run** against a cluster |
| Agent Engine memory adapter and agent (`ratchet/live/agent_engine.py`, `agent_engine/`) | **Not run**. Written from the public preview docs |

The offline demo uses rule-based stand-ins for the language model. Its scores
(4/15 to 14/15) show the mechanism working. They are not a measurement of how
a real model performs.

## Step 0. Run the demo (no accounts, no installs)

Needs Python 3.9 or later. On Windows use `python` instead of `python3`.

```
cd ratchet
python3 -m ratchet.run --pause
```

Press Enter to step through each attempt. Other options:

```
python3 -m ratchet.run --out run.json      # save the full log
python3 -m ratchet.run --rollback 2        # rebuild version 2 from its patches and re-score it
python3 -m unittest discover -s tests -v   # run the tests
```

## Step 1. What you are looking at

- **Version 0** is a plausible starting harness: default context sources, a
  write tool available, a tight tool-call budget.
- Each **attempt** is one proposed change with the evidence behind it.
- **ACCEPTED** means it fixed at least one held-out question and broke none.
- **REJECTED by the gate** means it broke something that worked. Nothing changes.
- **REFUSED by organization policy** means it asked for more than the
  organization allows. Nothing changes, and it is flagged for a human.

The editor only sees failures on the 15 training questions. The gate scores the
15 held-out questions, which the editor never sees.

## Step 2. Put it on Atlas (free cluster is enough)

1. In Atlas, create a free cluster and a database user.
2. Add your IP address under Network Access.
3. Click Connect, choose Drivers, and copy the connection string.

```
pip install pymongo
export MONGODB_URI="mongodb+srv://USER:PASSWORD@YOUR-CLUSTER.mongodb.net"
python3 -m ratchet.run --atlas --pause
```

Windows PowerShell: `$env:MONGODB_URI="..."`.

This loads the demo data into a database named `ratchet`, runs the agent's
aggregation pipelines on Atlas, and writes every harness version and attempt to
the `harness_versions` and `attempts` collections. Open them in the Atlas Data
Explorer during the demo. During evaluation, the delete tool is intercepted: it
counts what it would delete and records a violation, but deletes nothing.

## Step 3. Use a real model

```
pip install anthropic
export ANTHROPIC_API_KEY="..."
python3 -m ratchet.run --llm --max-attempts 3
```

Or `pip install openai` with `OPENAI_API_KEY`. Set `RATCHET_MODEL` to change the
model.

Cost warning: every attempt re-runs the held-out questions, and each question
can take several model calls. A full run is on the order of a thousand model
calls. Start with `--max-attempts 3`.

With a real model the story will differ from the scripted one. That is the
point, but rehearse it before presenting.

## Step 4. Atlas Agent Engine

Agent Engine is in public preview. This part is unverified; expect to adjust it.

Costs and prerequisites, from the docs: Docker Desktop, Python 3.11 or later,
and for memory a paid Atlas Flex cluster at minimum. The first deploy takes
roughly 10 to 20 minutes.

**4a. Tools and a first agent**

```
# download the CLI from https://agentengine.mongodb.com/download-cli, then:
agentengine auth login
agentengine create          # template: Hello World Agent, enable memory: yes
agentengine dev up          # runs it locally in Docker
```

**4b. Make it the Ratchet analyst.** In the project that `agentengine create`
made, replace the generated `main.py` and `agent.yaml` with the ones in
`agent_engine/ratchet-agent/`, and merge `agent_engine/project-config.yaml`
into the project's `project-config.yaml`. Read the note at the top of `main.py`.

**4c. Deploy**

```
agentengine init
agentengine atlas setup                      # provisions a paid Flex cluster by default
agentengine secret set ANTHROPIC_API_KEY
agentengine memory configure
agentengine deploy --auto
```

**4d. Keep rules and vocabulary in Agent Engine memory.** The loop talks to the
memory service directly so each user keeps their own vocabulary (a deployed
agent called by a service account shares one memory identity).

```
agentengine service-account create ratchet --project-id <project-id> --role AGENT_DEVELOPER
# exchange the client id and secret for a token as shown in the SDK README, then:
export AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN="..."
export AGENTIC_MEMORY_PROJECT_ID="<project-id>"
pip install agent-engine-sdk-memory
python3 -m ratchet.run --atlas --agent-engine-memory --pause
```

Candidate memory changes are staged locally and written to Agent Engine only
when the gate accepts them.

**4e. Policies.** In the platform UI, set the organization policy to match
`ORG_POLICY` in `ratchet/loop.py`. Apply each accepted project policy change
under Manage > Policies. Ratchet does not write policies through the API
because the request format was not confirmed from the docs. Policy changes take
about a minute to apply and affect only new executions.

## How the pieces map to Agent Engine

| Part of the harness | Lives in |
|---|---|
| Rules | Procedural memory, loaded by name |
| Shared vocabulary | Taxonomic memory (project-wide) |
| One user's vocabulary | Semantic memory (private to that user) |
| The editor's record of attempts | Episodic memory |
| Context policy, tool list, version lineage | `harness_versions` documents in Atlas |
| Tool access and budgets | Policy Engine project policies, capped by organization policy |

## Layout

```
ratchet/core.py        harness document, policy ceiling, memory
ratchet/world.py       dataset, aggregation pipelines, tools, test questions
ratchet/agents.py      analyst, editor, failure diagnosis
ratchet/loop.py        propose, ceiling check, gate, commit; rollback by replay
ratchet/run.py         command line
ratchet/live/          Atlas, model, and Agent Engine adapters
agent_engine/          deployable Agent Engine agent (starting point)
tests/                 test suite
```
