home = ENV["HOME"]
path_main = "/home/julien/Final-Project-Julien-Pascal"
cd(path_main)

include("src/simulated_annealing/samin.jl")

junk=2. # shows use of opj. fun. as a closure
function sse(x)
    objvalue = junk + sum(x.*x)
end


function rosenbrock_s(x)
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

k = 2
x = rand(k,1)
lb = [0.01, 0.01]
ub = [1.0, 3.0]

"""
# converge to global opt
xopt = samin(sse, x, lb, ub)
# no convergence within iteration limit
xopt = samin(sse, x, lb, ub, maxevals=10)
# initial point out of bounds
lb = 0.5*ub
x[1,1] = 0.2
xopt = samin(sse, x, lb, ub)
# optimum on bound of parameter space
x = 0.5 .+ 0.5*rand(k,1)
xopt = samin(sse, x, lb, ub, verbosity=1)
"""

xopt = samin(rosenbrock_s, x, lb, ub)

xopt[1]
xopt[2]