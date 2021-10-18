from time import time 

class Logger:
    def __init__(self):
        self.t_nn = 0
        self.t_sampling = 0
        self.t_mat = 0
        self.count = 0
        self.re = [] 
        self.te = []
        self.reGT = []
        self.t_meta = 0
    
    def increment(self):
        self.count += 1

    def record_re(self,e):
        self.re.append(e)
    
    def record_reGT(self,e):
        self.reGT.append(e)

    def record_te(self,e):
        self.te.append(e)

    def record_sampling(self,t):
        self.t_sampling += t

    def record_nn(self,t):
        self.t_nn += t 
    
    def record_meta(self,t):
        self.t_meta += t

    def record_mat(self,t):
        self.t_mat += t
    
    def avg_nn(self):
        return self.t_nn / self.count
    
    def avg_sampling(self):
        return self.t_sampling / self.count 
    
    def avg_mat(self):
        return self.t_mat / self.count

    def avg_all(self):
        return (self.t_mat+self.t_nn+self.t_sampling) / self.count
    
    def avg_meta(self):
        return self.t_meta / self.count 

    def recall(self,re_thres=15,te_thres=1.5):
        c = 0
        for i in range(len(self.re)):
            if self.re[i] <= re_thres and self.te[i] <= te_thres:
                c += 1
        return (c / self.count) * 100

