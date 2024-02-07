using LinearAlgebra

# Primitives
A = [0.4 0.1;
     0.7 0.2]

# Method one: direct inverse
B_inverse = inv(I - A)

# Method two: power series
function power_series(A)
    B_sum = zeros((2, 2))
    A_power = I
    for k in 1:50
        B_sum += A_power
        A_power = A_power * A
    end
    return B_sum
end

# Print maximal error
print(maximum(abs.(B_inverse - power_series(A))))
