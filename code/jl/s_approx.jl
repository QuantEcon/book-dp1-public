"""
Computes an approximate fixed point of a given operator T 
via successive approximation.

"""
function successive_approx(T,                  # operator (callable)
                           u_0;                # initial condition
                           tolerance=1e-6,     # error tolerance
                           max_iter=10_000,    # max iteration bound
                           print_step=25)      # print at multiples
    u = u_0
    error = Inf
    k = 1

    while (error > tolerance) & (k <= max_iter)
        
        u_new = T(u)
        error = maximum(abs.(u_new - u))

        if k % print_step == 0
            println("Completed iteration $k with error $error.")
        end

        u = u_new
        k += 1
    end

    if error <= tolerance
        println("Terminated successfully in $k iterations.")
    else
        println("Warning: hit iteration bound.")
    end

    return u
end
