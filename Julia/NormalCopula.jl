



type NormalCopula

	d     :: Int  # number of dimensions
	rho   :: Float64  # AR1 parameter
	sigma :: Array{Float64}  # sigma array

	# initiate copula with AR1 structure
	function NormalCopula(ndim::Int,rho::Float64)
 		m = abs(linspace(1.0,ndim,ndim) .- linspace(1.0,ndim,ndim)')
		sig = rho.^m
		return new(ndim,rho,sig)
	end
end	  

function show(io::IO, c::NormalCopula)
	print(io, "normal AR1 copula with $(c.d) dimensions and parameter $(c.rho)")
end



# random draws from the copula
function rnormCopula( c::NormalCopula, ndraw::Int )
	n = Normal()
	mn = MvNormal(c.sigma)
	pdf(n, rand(mn,ndraw))
end


# density of the copula
# TODO restrict u to be certain type?
function dnormCopula(u::Array{Float64}, c::NormalCopula)

	if length(size(u)) > 2
		error("u must be a matrix or a vector")
	end


	if ndims(u)>1
		if size(u,2) != c.d
			error("u must have $(c.d) columns")
		end
	else
		if length(u) != c.d
			error("u must have length $(c.d)")
		end
	end

	r = zeros(size(u,1))
	n = Normal(0,1)
	x = quantile(n,u)
	mn = MvNormal(c.sigma)
	r = logpdf(mn,x') .- sum(logpdf(n,x),2) 
	exp(r)
end
