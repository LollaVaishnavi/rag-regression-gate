# ADR 004: Run jitter is a declared input, not a measured one

## Status

Accepted.

## Context

The null the gate faces is not "nothing changed in theory" but "the same system was run twice". Two
runs of one retrieval stack do not return identical rankings: floating-point non-determinism, shard
and segment ordering, and near-tie breaks all move borderline documents between runs. That movement
is the noise a small golden set cannot tell apart from a real regression, so the null has to contain
it, or the whole project would be measuring against a null that is unrealistically quiet.

## Decision

Model run jitter as mean-zero Gaussian noise added to the similarity scores before ranking, with a
standard deviation set as a fraction of each query's score spread and driven by a per-run seed. Its
magnitude, `jitter_scale`, is a declared parameter of the retriever config, not a quantity measured
from a real system.

## Why a declared input is the honest choice here

The alternative would be to estimate a real jitter magnitude from a production retriever and present
it as measured. This repository has no production retriever, so any such number would be invented and
dressed as data, which is worse than a parameter that is openly a parameter. By making jitter a
declared input, every experiment states the noise level it assumed, and a reader can see that the
calibration result in ADR 005 is a property of the method at a stated noise level rather than a claim
about a specific real system.

## Consequences

The false-positive rates, the required-sample-size numbers, and the coin-flip near the threshold are
all conditional on `jitter_scale`. A real team would substitute their own observed run-to-run spread,
which the `power` command reads directly off their golden set without needing to model jitter at all.
The modelled jitter exists so the experiments have a realistic null to measure against, and nothing in
the shipped gate depends on the model being exactly right.
