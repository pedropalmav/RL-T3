# Tabular Methods 📋

## Experiment replication :test_tube:

There are multiple domains you can replicate: Simple Domains, Mutli Goal Domains, Multi Agent Domains and Partially Observable Domains.

### Simple Domains

```bash
python MainSimpleEnvs.py
```


### Multi Goal Domains :door:
To test this domain you have to run the following command:
```bash
python MainMultiGoal.py
```

Afterwards Q learning, Sarsa and 8-step Sarsa will run 30 times with 500 episode each, generating a graph with the average episode lengths.

### Multi Agent Domains 🐰🦊
These domains can be executed by running the command:
```bash
python MainMultiAgent.py
```

A menu will appear to select the type of setting you want to try: Centralized Cooperative, Decentralized Cooperative, and Decentralized Competitive; or to plot the average episode length of the settings. All settings run agents of Q learning to control the hunters, and prey for the last setting.

The execution times are:
- `Centralized Cooperative`: ~23 minutes
- `Decentralized Cooperative`: ~11 minutes
- `Decentralized Competitive`: ~18 minutes

>[!NOTE]
> We left the graph generation as a separate option because it would take so long to execute the three experiments consecutively. Instead, after the selected experiment is executed the results are saved on a json file on the `data/` folder.

### Partially Observable Domains 🔐
The next command will let you try this domain:
```bash
python MainPartiallyObservable.py
```

This will display a menu where you can select the type of memory you might want to add to the agents, the options are: No Memory, K-order memory, Binary memory, and K-order buffer. Then, the experiment will run Q learning, Sarsa and 16-step Sarsa 30 times with a 1000 episodes. Finally, a graph will be generated comparing the average episode length of each agent.

The execution times are:

- `No Memory`: ~14 minutes
- `K-order memory`: ~1 minutes
- `Binary memory`: ~3 minutes
- `K-order buffer`: ~30 seconds