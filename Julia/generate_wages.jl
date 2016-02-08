##########################################
# Create wages by value function iteration

path = "/home/julien/Documents/COURS/5A/MASTER THESIS/Labor Market/Julia/version 4" #the path to my directory

cd(path) #locate in the correct directory
require("wages2.jl")


tic()

transition = zeros(2*N,N);

for a = 1:M
	println(a)

	transition = calculate_W_low(a, N, M, discount, Markov, up_ini[:,:,a], down_ini[:,:,a], Z, Si_m);

    W_low[:,:,a] = transition[1:N,:];
	W_low_star[:,:,a] = transition[(N+1):(2*N),:];

end


jldopen(string(path,"/W_low_star.jld"), "w") do file
    write(file, "W_low_star", W_low_star) 
end

jldopen(string(path,"/W_low.jld"), "w") do file
    write(file, "W_low", W_low) 
end


transition2 = zeros(2*N,N);
for a = 1:M
	println(a)

	transition2 = calculate_W_high(a, N, M, discount, Markov, up_high_ini[:,:,a], down2_ini[:,:,a], Q, Si_m)

    W_high[:,:,a] = transition2[1:N,:];
    W_high_star[:,:,a] = transition2[(N+1):(2*N),:];

end
toc() #Takes 23 mn to run on my computer

jldopen(string(path,"/W_high_star.jld"), "w") do file
        write(file, "W_high_star", W_high_star) 
end

jldopen(string(path,"/W_high.jld"), "w") do file
       write(file, "W_high", W_high) 
end


        xgrid_plot = repmat(ygrid',N,1);
        ygrid_plot = repmat(ygrid,1,N);

        fig = figure("pyplot_surfaceplot",figsize=(10,10))
        ax = fig[:add_subplot](2,1,1) 
        cp = ax[:contour](xgrid_plot, ygrid_plot, W_low_star[:,:,100], colors="black", linewidth=2.0) 
        ax[:clabel](cp, inline=1, fontsize=10) 
        xlabel("X") 
        ylabel("Y")
        title("Contour Plot")

        subplot(212) 
        ax = fig[:add_subplot](2,1,2) 
        cp = ax[:contour](xgrid_plot, ygrid_plot, W_high_star[:,:,100], colors="black", linewidth=2.0) 
        ax[:clabel](cp, inline=1, fontsize=10) 
        xlabel("X") 
        ylabel("Y")
        title("Contour Plot")
