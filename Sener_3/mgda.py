import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MGDA_UB:
    itera = 250
    stop_crit = 1e-5

    def algo_1(v1v1, v1v2, v2v2):
        '''
        Implementation of Algorithm 1 in original paper: https://arxiv.org/abs/1810.04650
        Return: [gamma,cost]
        '''

        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
        elif v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
        else:
            gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
            cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def get_matrix_M(grads, grad_mat):
        '''
        Create Matrix M whose i,j value is the dot product of grad_z_i and grad_z_j where i and j are
        task.
        This funtion also calculate optimal gamma and optimal cost for 2 task problem.

        Return : [optimal task i, optimal task j, optimal gamma, optimal cost]
        '''

        cost_min = 1e8
        for i in range(len(grads)):
            # v1v1
            grad_mat[i,i] = torch.bmm(grads[i].view(grads[i].shape[0],1,grads[i].shape[1]),
                grads[i].view(grads[i].shape[0],grads[i].shape[1],1)).sum().data.cpu();

            for j in range(i+1,len(grads)):
                #v1v2
                grad_mat[i,j] = torch.bmm(grads[i].view(grads[i].shape[0],1,grads[i].shape[1]),
                    grads[j].view(grads[j].shape[0],grads[j].shape[1],1)).sum().data.cpu();
                grad_mat[j,i] = grad_mat[i,j]

                # v2v2
                grad_mat[j,j] = torch.bmm(grads[j].view(grads[j].shape[0],1,grads[j].shape[1]),
                    grads[j].view(grads[j].shape[0],grads[j].shape[1],1)).sum().data.cpu();
                
                gamma, cost = MGDA_UB.algo_1(grad_mat[i,i], grad_mat[i,j], grad_mat[j,j])
                if cost < cost_min:
                    cost_min = cost
                    sol = [i,j,gamma,cost]
        return sol


    def FW_solver(grads):
        ''' 
        Solve using FW solver
        Return : [Solution vector contains optimal gamma and 1-gamma, optimal cost]
        '''

        n=len(grads)
        grad_mat = np.zeros((n,n))
        init_sol = MGDA_UB.get_matrix_M(grads, grad_mat)

        sol_vec = np.zeros(n)
        sol_vec[init_sol[0]] = init_sol[2]
        sol_vec[init_sol[1]] = 1 - init_sol[2]

        # for n=2, initial solution is only optimal    
        if n < 3:
            return sol_vec , init_sol[3]

        
        itera_count = 0
        while itera_count < MGDA_UB.itera:
            v1 = np.dot(grad_mat, sol_vec)
            t = np.argmin(v1)

            v2 = grad_mat[:,t];

            v1v1 = np.dot(sol_vec, v1)
            v1v2 = np.dot(sol_vec, v2)
            v2v2 = grad_mat[t, t]

            gamma, cost = MGDA_UB.algo_1(v1v1, v1v2, v2v2)
            new_sol_vec = gamma*sol_vec
            new_sol_vec[t] += 1 - gamma

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MGDA_UB.stop_crit:
                return sol_vec, cost
            sol_vec = new_sol_vec

        return sol_vec, cost
