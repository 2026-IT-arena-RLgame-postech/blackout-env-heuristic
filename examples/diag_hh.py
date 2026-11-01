"""count how often hunt_hh fires (B5+hunt_hh vs opp), per match"""
import sys,collections
sys.path.insert(0,'.'); sys.path.insert(0,'examples')
import diag_refill as D
from blackout_env.heuristics import v34_bp
C=collections.Counter()
orig=v34_bp.StrategicHeuristicV34._count
def cnt(self,key):
    C[key]+=1; return orig(self,key)
v34_bp.StrategicHeuristicV34._count=cnt
from blackout_env.heuristics.params import resolve_descriptor as rd
ours=rd('B5')+',hunt_hh=30'
D.play(ours,sys.argv[1],list(range(8200000,8200012)),[i%2 for i in range(12)],T=1500)
print({k:v for k,v in C.items() if 'role' in k})
