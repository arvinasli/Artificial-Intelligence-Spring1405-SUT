# Arvin Baghal Asl - 403105793

"""Probability models - Bayesian Networks"""

from functools import reduce
import random
import numpy as np
from inspect import getsource


class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, var_name='?', freq=None):
        """If freq is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized."""
        self.prob = {}
        self.var_name = var_name
        self.values = []
        if freq:
            for (v, p) in freq.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        """Given a value, return P(value)."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        """Set P(val) = p."""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('{}: ' + numfmt).format(v, p) for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({})".format(self.var_name)

# Helper Functions

def event_values(event, variables):
    """Return a tuple of the values of variables in event.
    >>> event_values({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])

def extend(s, var, val):
    """Copy dict s and extend it by setting var to val; return copy."""
    return {**s, var: val}

def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)

def psource(*functions):
    """Print the source code for the given function(s)."""
    source_code = '\n\n'.join(getsource(fn) for fn in functions)
    
    print(source_code)


# Bayesian Network Core Classes

class BayesNode:
    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        """X is a variable name, and parents a sequence of variable
        names or a space-separated string. cpt, the conditional
        probability table, takes one of these forms:

        * A number, the unconditional probability P(X=true). You can
          use this form when there are no parents.

        * A dict {v: p, ...}, the conditional probability distribution
          P(X=true | parent=v) = p. When there's just one parent.

        * A dict {(v1, v2, ...): p, ...}, the distribution P(X=true |
          parent1=v1, parent2=v2, ...) = p. Each key must have as many
          values as there are parents. You can use this form always;
          the first two are just conveniences.

        In all cases the probability of X being false is left implicit,
        since it follows from P(X=true).

        >>> X = BayesNode('X', '', 0.2)
        >>> Y = BayesNode('Y', 'P', {T: 0.2, F: 0.7})
        >>> Z = BayesNode('Z', 'P Q',
        ...    {(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
        """
        if isinstance(parents, str):
            parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value: bool, event: dict):
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""
        # Return P(X=value | parents). Use self.cpt and event_values.
        # If value is True, return the probability from cpt.
        # If value is False, return 1 - that probability.
        event_vals = event_values(event, self.parents)

        p = self.cpt[event_vals]

        if value:
            return p
        return 1 - p

    def sample(self, event: dict):
        """Sample from the distribution for this variable conditioned
        on event's values for parent_variables. That is, return True/False
        at random according with the conditional probability given the
        parents."""
        # Return True with probability P(self.variable=True | event).
        # Use self.p() and the probability() helper.
        p = self.p(True, event)
        return probability(p)

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))


class BayesNet:
    """Bayesian network containing only boolean-variable nodes."""

    def __init__(self, node_specs=None):
        """Nodes must be ordered with parents before children."""
        self.nodes: list[BayesNode] = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        # Unpack node_spec into a BayesNode and add it to self.nodes.
        # Update self.variables and the children list of each parent.
        X, parents, cpt = node_spec
        if X in self.variables:
            raise ValueError(f"Variable '{X}' already exists in the network!")
        
        new_node = BayesNode(X, parents, cpt)
        self.nodes.append(new_node)

        self.variables.append(X)
        for parent in new_node.parents:
            parent_node = self.variable_node(parent)
            if parent_node:
                parent_node.children.append(new_node)

    def variable_node(self, var):
        """Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        # Find and return the BayesNode with self.variable == var.
        for node in self.nodes:
            if node.variable == var:
                return node
            
        return None

    def variable_values(self, var):
        """Return the domain of var."""
        return [True, False]

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)

# Exact Inference

def enumeration_ask(X, e, bn: BayesNet):
    """Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn.
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        # Compute P(X=xi | e) using enumerate_all.
        # Hint: Extend e with X=xi and pass all variables of bn.
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()


def enumerate_all(variables, e, bn: BayesNet):
    """Return the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than variables). Parents must precede children in variables."""
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.variable_node(Y)
    if Y in e:
        # Y is an evidence variable. Return its probability multiplied by the rest.
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        # Y is a hidden variable. Sum over all its possible values.
        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))


# Variable Elimination
def elimination_ask(X, e, bn: BayesNet):
    """Compute bn's P(X|e) by variable elimination.
    >>> elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    factors = []
    for var in reversed(bn.variables):
        # Create a factor for this variable, then eliminate it if hidden.
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)
    return pointwise_product(factors, bn).normalize()


def is_hidden(var, X, e):
    """Is var a hidden variable when querying P(X|e)?"""
    # Return True if var is neither the query nor an evidence variable.
    return var not in X and var not in e


def make_factor(var, e, bn: BayesNet):
    """Return the factor for var in bn's joint distribution given e."""
    node = bn.variable_node(var)
    # Collect variables (var + parents) that are not in evidence.
    variables = [X for X in [var] + node.parents if X not in e]
    # Build cpt by iterating over all events for these variables.
    cpt = {event_values(e1, variables): node.p(e1[var], e1)
           for e1 in all_events(variables, bn, e)}
    return Factor(variables, cpt)


def pointwise_product(factors, bn: BayesNet):
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)


def sum_out(var, factors, bn: BayesNet):
    """Eliminate var from all factors by summing over its values."""
    result, var_factors = [], []
    for f in factors:
        # Separate factors that contain var from those that don't.
        (var_factors if var in f.variables else result).append(f)
    # Multiply factors containing var, then sum out var.
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result


class Factor:
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwise_product(self, other, bn: BayesNet):
        """Multiply two factors, combining their variables."""
        # Compute union of variables from both factors.
        variables = list(set(self.variables) | set(other.variables))
        # Build cpt by multiplying probabilities for each event.
        cpt = {event_values(e, variables): self.p(e) * other.p(e)  for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def sum_out(self, var, bn: BayesNet):
        """Make a factor eliminating var by summing over its values."""
        # Remove var from the variable list.
        variables = [X for X in self.variables if X != var]
        # Sum probabilities over all values of var.
        cpt = {event_values(e, variables): sum(self.p(extend(e, var, val))
               for val in bn.variable_values(var))
               for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def normalize(self):
        """Return my probabilities; must be down to one variable."""
        assert len(self.variables) == 1
        return ProbDist(self.variables[0], {k: v for ((k,), v) in self.cpt.items()})

    def p(self, e):
        """Look up my value tabulated for e."""
        # Return the probability from self.cpt for the given event.
        return self.cpt[event_values(e, self.variables)]


def all_events(variables, bn: BayesNet, e):
    """Yield every way of extending e with values for all variables."""
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        # Recursively extend e with each possible value of X.
        for e1 in all_events(rest, bn, e):
            for x in bn.variable_values(X):
                yield extend(e1, X, x)

# Approximate Inference (Sampling Methods)

def prior_sample(bn: BayesNet):
    """Randomly sample from bn's full joint distribution.
    The result is a {variable: value} dict."""
    # Implement this function.
    # Iterate over bn.nodes in order. For each node, sample a value
    # given the values already assigned to its parents in event.
    sample = {}
    for node in bn.nodes:
        sample[node.variable] = node.sample(sample)
    
    return sample


def rejection_sampling(X, e, bn: BayesNet, N=10000):
    """Estimate the probability distribution of variable X given
    evidence e in BayesNet bn, using N samples.
    Raises a ZeroDivisionError if all the N samples are rejected.
    >>> random.seed(47)
    >>> rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.7, True: 0.3'
    """
    # Implement rejection sampling.
    # 1. Generate N prior samples.
    # 2. Keep only those consistent with evidence e.
    # 3. Count how often X takes each value.
    # 4. Return a ProbDist over X.
    consistent_samples = []
    for _ in range(N):
        sample = prior_sample(bn)
        if consistent_with(sample, e):
            consistent_samples.append(sample)

    if len(consistent_samples) == 0:
        raise ZeroDivisionError("All the N samples are rejected!")   
    
    counts_X = {True: 0, False: 0}
    for sample in consistent_samples:
        counts_X[sample[X]] += 1
    
    return ProbDist(X, counts_X)


def consistent_with(event, evidence):
    """Is event consistent with the given evidence?"""
    # Implement this function.
    # Return True iff for every var in evidence, event[var] == evidence[var].
    for var in evidence.keys():
        if event[var] != evidence[var]:
            return False
        
    return True


def likelihood_weighting(X, e, bn: BayesNet, N=10000):
    """Estimate the probability distribution of variable X given
    evidence e in BayesNet bn.
    >>> random.seed(1017)
    >>> likelihood_weighting('Burglary', dict(JohnCalls=T, MaryCalls=T),
    ...   burglary, 10000).show_approx()
    'False: 0.702, True: 0.298'
    """
    # Implement likelihood weighting.
    # 1. Generate N weighted samples (use weighted_sample).
    # 2. For each sample, accumulate its weight into the count for sample[X].
    # 3. Return a ProbDist over X.
    weights_X = {True: 0, False: 0}
    for _ in range(N):
        sample, w = weighted_sample(bn, e)
        weights_X[sample[X]] += w
    
    return ProbDist(X, weights_X)


def weighted_sample(bn: BayesNet, e):
    """Sample an event from bn that's consistent with the evidence e;
    return the event and its weight."""
    # Implement this function.
    # 1. Start with event = dict(e) and weight w = 1.
    # 2. For each node in topological order:
    #    - If node is in e: multiply w by P(node=evidence_value | event).
    #    - If node is not in e: sample its value from P(node | event).
    # 3. Return (event, w).
    event = dict(e)
    w = 1.0
    for node in bn.nodes:
        if node.variable in e:
            w *= node.p(e[node.variable], event)
        else:
            event[node.variable] = node.sample(event)

    return (event, w)


def gibbs_ask(X, e, bn: BayesNet, N=1000):
    """Estimate the probability distribution of variable X given
    evidence e in BayesNet bn using Gibbs sampling."""
    assert X not in e, "Query variable must be distinct from evidence"
    # Implement Gibbs sampling.
    # 1. Initialize state with evidence e and random values for other variables.
    # 2. For N iterations:
    #    - For each non-evidence variable Zi, resample it from markov_blanket_sample.
    #    - Increment the count for the current value of X.
    # 3. Return a ProbDist over X.
    state = dict(e)
    Z = [node.variable for node in bn.nodes if node.variable not in e]
    for var in Z:
        state[var] = random.choice([True, False])

    counts_X = {True: 0, False: 0}
    for _ in range(N):
        for Zi in Z:
            state[Zi] = markov_blanket_sample(Zi, state, bn)
            
        counts_X[state[X]] += 1
    
    return ProbDist(X, counts_X)


def markov_blanket_sample(X, e, bn: BayesNet):
    """Return a sample from P(X | mb) where mb denotes that the
    variables in the Markov blanket of X take their values from event
    e (which must assign a value to each). The Markov blanket of X is
    X's parents, children, and children's parents."""
    # Implement Markov blanket sampling.
    # 1. Build a ProbDist Q over the values of X.
    # 2. For each possible value xi of X:
    #    - Extend e with X=xi.
    #    - Compute P(xi | parents(X)) * product over children Y of P(y | parents(Y)).
    # 3. Return True with probability proportional to Q[True].
    Xnode = bn.variable_node(X)
    Q = {True: 0.0, False: 0.0}
    
    for xi in bn.variable_values(X):
        event = dict(e)
        event[X] = xi
        
        # calculate P(xi | parents(X))
        p = Xnode.p(xi, event)
        
        # product over all children Y of P(y | parents(Y))
        children = [node for node in bn.nodes if X in node.parents]
        for Ynode in children:
            p *= Ynode.p(event[Ynode.variable], event)
            
        Q[xi] = p
        
    p_true = Q[True] / (Q[True] + Q[False])

    return probability(p_true)