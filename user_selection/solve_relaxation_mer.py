import cvxpy as cp
import numpy as np
import time
import numpy.linalg as LA

def solve_relaxed(H=None, gamma_l=None, gamma_u=None, powe_db=1.0, sigma=1.0, rho=1.0):
    # define problem parameters
    N_t, K = H.shape

    P_T = np.power(10, powe_db/10)
    b_k = [P_T*LA.norm(H[:, k])**2 for k in range(K)]
    for k in range(K):
        if gamma_l[k]>0:
            aa = b_k[k]/gamma_l[k] -sigma
            b_k[k] = np.minimum(b_k[k], aa)


    # define optimization variables
    R_X  = cp.Variable((N_t,N_t), hermitian=True)
    Gamma = cp.Variable(K)
    W = [cp.Variable((N_t,N_t), hermitian= True) for _ in range(K)]
    s = cp.Variable(K)
    Z = cp.Variable((N_t, N_t), hermitian=True)
    # define the optimization objective
    obj = -cp.sum(cp.log(1 + Gamma)) +rho * cp.real(cp.trace(Z))

    # define the constraints
    constraints = [
        cp.real(cp.trace(R_X)) <= P_T,
        R_X >> cp.sum(W),
        # R_X >> 0,
        cp.bmat([[Z, np.eye(N_t)], [np.eye(N_t), R_X]]) >> 0,
    ]
    for k in range(K):
        Q_k = np.outer(H[:,k],np.conj(H[:,k]))
        constraints +=[
            W[k] >> 0,
            cp.real(cp.trace(Q_k @ W[k]) -s[k]) >= Gamma[k] * sigma,
            cp.real(cp.trace(Q_k @ (R_X - W[k]))) <= b_k[k],
            cp.real(cp.trace(Q_k @ (R_X - W[k]))) >= 0,
            s[k] >= gamma_l[k] * cp.real(cp.trace(Q_k @ (R_X - W[k]))),
            #s[k] >= gamma_u[k] * cp.real(cp.trace(Q_k @ (R_X - W[k]))) + (Gamma[k]-gamma_u[k])*b_k[k],
            #s[k] <= gamma_u[k]* cp.real(cp.trace(Q_k @ (R_X - W[k]))),
            #s[k] <= (Gamma[k]-gamma_l[k])*b_k[k] + gamma_l[k] * cp.real(cp.trace(Q_k @ (R_X - W[k]))),
        ]
    constraints +=[
        Gamma >=gamma_l,
        Gamma <= gamma_u,
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        # mosek_log = prob.solver_stats.mosek
        # print("MOSEK Log:", mosek_log)
    except Exception as e:
        print(e)
       # return np.zeros(K), np.zeros((N_t, N_t)), np.inf, False
    #
    # if prob.status in ['infeasible', 'unbounded']:
    #     # print('infeasible antenna solution')
    #     return np.zeros(N_t), np.zeros((N_t, K)), np.inf, False
    optimal_R_X = R_X.value
    optima_Gamma = Gamma.value
    optimal_W = [W[k].value for k in range(K)]
    optimal_s = s.value

    optimal_Z = Z.value
    optimal_objective = prob.value
    #optimal_objective = -np.sum(np.log(1 + optima_Gamma)) + rho * np.real(np.trace(optimal_Z))
    if prob.status == cp.OPTIMAL:

        feas = 1
    else:

        feas = -1




    return optima_Gamma, optimal_W, optimal_R_X, optimal_s, optimal_objective, feas

def initial_gamma_u(H=None, powe_db=1.0, sigma=1.0):

    N_t, K = H.shape

    P_T = np.power(10, powe_db/10)

    for k in range(K):
        Q_k = np.outer(H[:,k],np.conj(H[:,k]))

    gamma_u = np.zeros(K)
    for k in range(K):
        R_X = cp.Variable((N_t, N_t), hermitian=True)
        W = [cp.Variable((N_t, N_t), hermitian=True) for _ in range(K)]
        obj = cp.real(cp.trace(Q_k @ W[k])/(cp.trace(Q_k @ (R_X)) + sigma))
        constraints = [
            cp.real(cp.trace(R_X)) <= P_T,
            R_X >> sum(W),
        ]
        for i in range(K):
            constraints += [
                W[i] >> 0]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        try:
            prob.solve(solver=cp.MOSEK, verbose=True)
        except Exception as e:
            print(e)
        gamma_u[k] = prob.value
    return gamma_u
if __name__=='__main__':
    import numpy.linalg as LA
    import time
    np.random.seed(1)
    N_t, K = 5, 4
    H = (np.random.randn(N_t, K) + 1j * np.random.randn(N_t, K)) / np.sqrt(2)
    powe_db = 10
    sigma = 5
    rho = 1
    gamma_l = np.zeros(K)
    P_T = np.power(10, powe_db / 10)
    #gamma_u_new = initial_gamma_u(H, powe_db=powe_db, sigma=sigma)
    gamma_u = [P_T*LA.norm(H[:,k])**2/sigma for k in range(K)]



    t1 = time.time()
    Gamma, W, R_X, s, obj, feas = solve_relaxed(H=H, gamma_l=gamma_l, gamma_u=gamma_u, powe_db = powe_db, sigma=sigma, rho=rho)
    time_taken = time.time()-t1
    w, v = np.linalg.eig(R_X)
    #print(W)
    print(w)
    print(feas)
    print('time taken', time_taken)






