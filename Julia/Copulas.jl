

module Copulas

# dependencies

import Distributions: Normal, MvNormal, pdf, logpdf, quantile, rand
import Base.show

# exports
export NormalCopula, dnormCopula, rnormCopula

# loading
include("NormalCopula.jl")


end	# module
