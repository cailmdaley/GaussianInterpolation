using GaussianInterpolation, CMBLensing
using Test


@test size(coords(zeros(3,2))) == (6,2)
@test size(coords(FlatMap(rand(3,3)))) == (9, 2)
