 # vector obs
 len : 96

## part 1 - all agents info
10 * (2+1+6) for all agents

for each agents:
(x, y) + (1 if ally else -1) + (holding item ont hot)

## part 2 - current agent class info
and + 3 for each agent

## part 3 - scores, time
and + 3 for all agents


for input preprocessing

```python
obs = {} # dict
positions = []
team = []
items = []
clazz = []
scores = []
time_left = 1
for i in range(10):
    positions.append(obs["agent_0"]["vector"][i*9:i*9+2])
    team.append(obs["agent_0"]["vector"][i*9+2])
    items.append(obs["agent_0"]["vector"][i*9+3:i*9+9])
    clazz.append(obs[f"agent_{i}"]["vector"][90:93])
    
scores = obs["agent_0"]["vector"][93:95]
time_left = obs["agent_0"]["vector"][95]
```


