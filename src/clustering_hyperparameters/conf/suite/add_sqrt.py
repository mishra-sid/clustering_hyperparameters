from yaml import load, dump, Loader, Dumper
from math import sqrt, ceil

x = None
with open('nlp.yaml') as f:
    x = load(f, Loader=Loader)
    for ind in range(len(x['datasets'])): 
        x['datasets'][ind]['metadata']['max_num_clusters'] = int(ceil(sqrt(x['datasets'][ind]['metadata']['num_instances']))) 

with open('nlp.yaml', 'w') as wf:
    out = dump(x, wf, Dumper=Dumper, default_flow_style=False)

