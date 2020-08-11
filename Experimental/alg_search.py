import numpy as np


class gen_alg:
    
    
    
    def __init__(self,space, pop_size=10, mut_rate=0.03, cross_rate=0.4, use_tour=True, use_elit=True ):
        self.space = space
        self.n_features = len(list(self.space.keys()))
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.use_tour = use_tour
        self.use_elit = use_elit
        
        n_variables = len(self.space.keys())
        self.pop = np.zeros((pop_size,n_variables))
        for i in range(self.pop_size):
            self.pop[i,:] = np.array([np.random.choice(self.space[j], 1) for j in self.space.keys()])[:,0]
        #self.pop = initiate_pop()
    
    def initiate_pop(self):
        n_variables = len(self.space.keys())
        self.pop = np.array((pop_size,n_variables))
        for i in range(self.pop_size):
            self.pop[i,:] = np.array([np.chose(self.space[j]) for j in self.space.keys()])

    def mutate(self):
        
        mut_mat = np.argwhere(np.random.rand(self.pop.shape[0], self.pop.shape[1]) <  self.mut_rate)
        for i in mut_mat:
            k = list(self.space.keys())
            
            index = np.where(np.array(self.space[k[i[1]]]) == self.pop[i[0],i[1]])
            
            if index[0][0] == len(self.space[k[i[1]]])-1:
                self.pop[i[0],i[1]] = self.space[k[i[1]]][index[0][0]-1]
            elif index[0][0] == 0:
                self.pop[i[0],i[1]] = self.space[k[i[1]]][index[0][0]+1]
            else:
                change = 1 if np.random.rand() < 0.5 else -1
                self.pop[i[0],i[1]] = self.space[k[i[1]]][index[0][0] + change]
               
    
    def cross_over(self, res):

        res = (1/res)/np.sum(1/res)
        num_parents = 2 * self.pop_size-2
        arr = np.arange(self.pop_size)
        best = self.pop[np.argmax(res),:]
        parent_vec = np.random.choice(arr, size = num_parents, replace = True, p = res)
        for i in range(0,num_parents,2):        
            if np.random.rand() < self.cross_rate:
                split = np.random.randint(1,self.n_features)
                tmp = self.pop[parent_vec[i],:split]
                self.pop[parent_vec[i],:split] = self.pop[parent_vec[i+1],:split]
                self.pop[parent_vec[i+1],:split] = tmp
        self.pop[-1,:] = best
    def print_best(self, res):
        tmp = res
        res = (1/res)/np.sum(1/res)
        arg = np.argmax(res)
        print('Val_loss: ',tmp[arg],'\n',self.pop[arg,:])