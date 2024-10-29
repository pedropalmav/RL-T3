# Tabular Methods 📋

## Experiment replication :test_tube:

There are multiple domains you can replicate: Simple Domains, Mutli Goal Domains, Multi Agent Domains and Partially Observable Domains.

### Multi Goal Domains :door:
To test this domain you have to run the following command:
```bash
python MainMultiGoal.py
```

### Multi Agent Domains 🐰🦊
These domains can be executed by running the command:
```bash
python MainMultiAgent.py
```

### Partially Observable Domains 🔐
The next command will let you try this domain:
```bash
python MainPartiallyObservable.py
```

This will display a menu where you can select the type of memory you might want to add to the agents, the options are: No Memory, K-order memory, Binary memory, and K-order buffer. Then, the experiment will run Q learning, Sarsa and 16-step Sarsa 30 times with a 1000 episodes. Finally, a graph will be generated comparing the average episode length of each agent.