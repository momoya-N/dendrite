function SH_fourier_output(L, N, p, t_max, Nframe, folder_index)
    # ϕ_tilde[:, k_ϕ_max]を保存する
    folder_name = Folder_name(L, N, p, t_max, Nframe, folder_index)
    q_ϕ, q_ψ, ε, χ, α = p
    ϕ_data = readdlm(folder_name * "/phi-$q_ϕ-$q_ψ-$ε-$χ-$α-$L-$N-$t_max-$Nframe.dat")
    # ψ_data = readdlm(folder_name * "/psi-$q_ϕ-$q_ψ-$ε-$χ-$α-$L-$N-$t_max-$Nframe.dat")
    T = readdlm(folder_name * "/T-$q_ϕ-$q_ψ-$ε-$χ-$α-$L-$N-$t_max-$Nframe.dat")[1:end]

    Δx = L / N
    Δt = T[2] - T[1]
    start_frame = 2 * Nframe ÷ 10

    ϕ_tilde = fft(ϕ_data[start_frame:Nframe, :]) ./ Nframe ./ N
    ψ_tilde = fft(ψ_data[start_frame:Nframe, :]) ./ Nframe ./ N

    N_t = size(ϕ_tilde)[1]
    T_tot = t_max * (Nframe - start_frame) / Nframe

    # T = range(dt*Nstep*start_frame/Nframe, Nstep*dt, length=N_t)
    # X = range(0,  L, length=N)
    if N % 2 == 0
        k = [0:N÷2; -N÷2+1:-1] * (2π / L)
    else
        k = [0:N÷2; -N÷2:-1] * (2π / L)
    end

    if N_t % 2 == 0
        ω = [0:N_t÷2; -N_t÷2+1:-1] * (2π / (T_tot))
    else
        ω = [0:N_t÷2; -N_t÷2:-1] * (2π / (T_tot))
    end

    ϕ_tilde = ϕ_tilde[sortperm(ω), :]
    ϕ_tilde = ϕ_tilde[:, sortperm(k)]

    ψ_tilde = ψ_tilde[sortperm(ω), :]
    ψ_tilde = ψ_tilde[:, sortperm(k)]

    ω = sort(ω)
    k = sort(k)

    # ω_ϕ_max, k_ϕ_max = Tuple(argmax(abs.(ϕ_tilde)))
    # ω_ψ_max, k_ψ_max = Tuple(argmax(abs.(ψ_tilde)))

    k_ϕ_max = N ÷ 2 + 1
    ϕ_tilde_kmax = ϕ_tilde[:, k_ϕ_max]

    # output
    open("$folder_name/phi_tilde_kmax.dat", "w") do f
        for i in 1:N_t
            print(f, @sprintf("%.6g ", abs(ϕ_tilde_kmax[i])^2))
        end
        println(f)
    end
end