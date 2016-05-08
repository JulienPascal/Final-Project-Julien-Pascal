module Test

using FactCheck, PyPlot, Distributions, StatsFuns, JLD, HDF5, DataFrames, Copulas, NLopt, BlackBoxOptim

include("test.jl")

FactCheck.exitstatus()


end