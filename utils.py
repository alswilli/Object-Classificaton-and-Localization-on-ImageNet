
"""
Takes a list (l) and returns a list of evenly sized chunks, size n. Note the last list in the returned list
may have size <=n. 
"""

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]