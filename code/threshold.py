"""Fixed-population sign certificates from the two exclusive queue strata."""
from functools import lru_cache

import numpy as np
from scipy.stats import hypergeom

@lru_cache(maxsize=40000)
def intervals_for_counts(N, n, counts, tail):
    """Invert both exact tails, keeping the full integer feasible range."""
    counts = np.asarray(counts, dtype=int)
    if n == 0:
        return np.zeros_like(counts), np.full_like(counts,N)
    if n == N:
        return counts,counts
    feasible_low,feasible_high=counts,counts+N-n
    lower,upper=feasible_low.copy(),feasible_high.copy()
    while (lower<upper).any():
        pivot=(lower+upper)//2
        admissible=hypergeom.sf(counts-1,N,pivot,n)>=tail
        lower=np.where(admissible,lower,pivot+1)
        upper=np.where(admissible,pivot,upper)
    lo=lower
    lower,upper=feasible_low.copy(),feasible_high.copy()
    while (lower<upper).any():
        pivot=(lower+upper+1)//2
        admissible=hypergeom.cdf(counts,N,pivot,n)>=tail
        lower=np.where(admissible,pivot,lower)
        upper=np.where(admissible,upper,pivot-1)
    return lo,lower

def bounds(N,n,x,tail):
    counts,idx=np.unique(x,return_inverse=True)
    lo,hi=intervals_for_counts(int(N),int(n),tuple(map(int,counts)),tail)
    return lo[idx],hi[idx]

def allocation(sizes, check_totals):
    sizes=np.asarray(sizes,dtype=int)
    seen=np.zeros(len(sizes),dtype=int)
    n=int(sizes.sum())
    active=sizes>0
    checkpoints=[]
    for t in range(1,check_totals[-1]+1):
        allowed=(seen<sizes)
        uncovered=active & (seen==0)
        if uncovered.any():
            allowed=uncovered
        side=np.argmax(np.where(allowed,t*sizes/n-seen,-np.inf))
        seen[side]+=1
        if t in check_totals:
            checkpoints.append(seen.copy())
    assert (np.asarray(checkpoints).sum(1)==check_totals).all()
    return checkpoints

def strict_sign(lo,hi):
    return np.where(lo>0,1,np.where(hi<0,-1,0)).astype(np.int8)

def simulate(known, sizes, successes, cap, looks, alpha, rng, draws):
    # Stratum 0 contributes to the default; stratum 1 to the challenger.
    sizes=np.asarray(sizes,dtype=int)
    successes=np.asarray(successes,dtype=int)
    assert (0<=successes).all() and (successes<=sizes).all()
    total=known+int(successes[1]-successes[0])
    sharp=(known-int(sizes[0]),known+int(sizes[1]))
    sign=int(strict_sign(*sharp))
    cap=min(int(cap),int(sizes.sum()))
    if sign or cap==0:
        return total,sharp,[(np.zeros(draws,dtype=int),np.full(draws,sign,dtype=np.int8)) for _ in range(3)]
    totals=sorted(set(min(cap,max(1,round(f*cap))) for f in looks))
    assert totals[-1]==cap
    allocation_by_look=allocation(sizes,totals)
    active=max(1,int((sizes>0).sum()))
    tail=alpha/(2*active*len(totals))
    x=np.zeros((2,draws),dtype=int)
    seen=np.zeros(2,dtype=int)
    stop=np.full(draws,cap,dtype=int)
    signs=np.zeros(draws,dtype=np.int8)
    stopped=np.zeros(draws,dtype=bool)
    for cost,counts in zip(totals,allocation_by_look):
        for i in range(2):
            k=int(counts[i]-seen[i])
            if k:
                good=successes[i]-x[i]
                bad=sizes[i]-seen[i]-good
                x[i]+=rng.hypergeometric(good,bad,k)
        seen=counts
        l0,u0=bounds(sizes[0],seen[0],x[0],tail)
        l1,u1=bounds(sizes[1],seen[1],x[1],tail)
        this=strict_sign(known+l1-u0,known+u1-l0)
        first=(this!=0)&~stopped
        stop[first]=cost
        signs[first]=this[first]
        stopped|=first
    terminal_tail=alpha/(2*active)
    l0,u0=bounds(sizes[0],seen[0],x[0],terminal_tail)
    l1,u1=bounds(sizes[1],seen[1],x[1],terminal_tail)
    single=strict_sign(known+l1-u0,known+u1-l0)
    fixed_cost=np.full(draws,cap,dtype=int)
    return total,sharp,[(stop,signs),(fixed_cost,single),(fixed_cost,this)]

def stratum_counts(a,c,observed,truth):
    first=a&~c
    second=c&~a
    known=int((second&observed&truth).sum())-int((first&observed&truth).sum())
    unknown=(first&~observed,second&~observed)
    sizes=[int(x.sum()) for x in unknown]
    successes=[int((x&truth).sum()) for x in unknown]
    return known,sizes,successes
