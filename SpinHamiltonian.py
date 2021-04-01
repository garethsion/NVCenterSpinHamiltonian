import numpy as np
import qutip as qt
import math

class SpinHamiltonian:
    def __init__(self):
        self.sx = qt.jmat(1,'x')
        self.sy = qt.jmat(1,'y')
        self.sz = qt.jmat(1,'z')

        self.ge = 28e09 # Gyromagnetic ratio of NV center
        self.D = 2.87e09 # Zero field splitting

        self.zero_field = self.D*((self.sz*self.sz)-(2/3)*qt.qeye(3))
    
    def interaction(self):
        pass

    def transition_frequencies(self):
        pass


class ZeemanSpinHamiltonian(SpinHamiltonian):
    def __init__(self):
        SpinHamiltonian.__init__(self)
        self.ms0 = 0 # Energy of ms=0 state initialized to 0

    def interaction(self,B,theta,z):
        if z:
            Bx = 0
            By = 0
            Bz = B
        else:
            Bx = B * math.cos(theta) # calculating B from its magnitude, polar angle
            By = B * math.sin(theta)
            Bz = 0
        
        Hzee = self.ge*(Bz*self.sz + Bx*self.sx + By*self.sy)
        return Hzee

    def transition_frequencies(self, B, theta, z ):
        H = self.zero_field + self.interaction(B,theta,z)
        estates = H.eigenstates()
        egvals = estates[0]

        if(B == 0): self.ms0 = egvals[0] 
        f1 = egvals[2] - self.ms0 if(z) else egvals[2]-egvals[0] # to distinguish parallel and perpendiculr energies as qutip sorts them
        f0 = abs(egvals[1] + egvals[0] - (2*self.ms0)) if(z) else egvals[1]-egvals[0] # absolute value of frequency

        return np.array([f1,f0])

    def spin_projection(self, B, theta=0, z=0, *args, **kwargs):

        H = self.interaction(B,theta,z)
        H = self.zero_field + H

        egst = H.eigenstates()

        msx0 = qt.expect(self.sx,egst[1][0])   # spin projection of 0th spin eigenstate of Hs on x axis
        msy0 = qt.expect(self.sy,egst[1][0])   # spin projection of 0th spin eigenstate of Hs on y axis
        msz0 = qt.expect(self.sz,egst[1][0])   # spin projection of 0th spin eigenstate of Hs on z axis

        msx1 = qt.expect(self.sx,egst[1][1])   # spin projection of 1st spin eigenstate of Hs on x axis
        msy1 = qt.expect(self.sy,egst[1][1])   # spin projection of 1st spin eigenstate of Hs on y axis
        msz1 = qt.expect(self.sz,egst[1][1])   # spin projection of 1st spin eigenstate of Hs on z axis

        msx2 = qt.expect(self.sx,egst[1][2])   # spin projection of 2nd spin eigenstate of Hs on x axis
        msy2 = qt.expect(self.sy,egst[1][2])   # spin projection of 2nd spin eigenstate of Hs on y axis
        msz2 = qt.expect(self.sz,egst[1][2])   # spin projection of 2nd spin eigenstate of Hs on z axis

        return np.array([msx0,msy0,msz0,msx1,msy1,msz1,msx2,msy2,msz2])

class HyperfineSpinHamiltonian(SpinHamiltonian):
    def __init__(self):
        SpinHamiltonian.__init__(self)
        self.gc = 10.705e6                  # gyromagnetic ratio of C-13 nucleus in Hz/T for hyperfine interaction

        self.Axx = 189.3e6
        self.Ayy = 128.4e6
        self.Azz = 128.9e6
        self.Axz = 24.1e6                   # Hyperfine Tensor components in NV frame of reference. Taken from reference 2.
        self.Ix = qt.jmat(1,'x')
        self.Iy = qt.jmat(1,'y')
        self.Iz = qt.jmat(1,'z')            # Spin 1/2 operators for C-13 nucleus

        self.comp1 = self.Axx*qt.tensor(self.sx,self.Ix)
        self.comp2 = self.Ayy*qt.tensor(self.sy,self.Iy)
        self.comp3 = self.Azz*qt.tensor(self.sz,self.Iz)
        self.comp4 = self.Axz*(qt.tensor(self.sx,self.Iz)+qt.tensor(self.sz,self.Ix))
        self.Hhf = self.comp1 + self.comp2 + self.comp3 + self.comp4

    def eigenvalues(self,Bz):
        H = self.D*(qt.tensor(self.sz*self.sz,qt.qeye(3))-(2/3)*qt.tensor(qt.qeye(3),qt.qeye(3))) + self.ge*Bz*qt.tensor(self.sz,qt.qeye(3)) + self.gc*Bz*qt.tensor(qt.qeye(3),self.Iz) + self.Hhf  
        return H.eigenstates()[0]

    def transitionFreqs(self,Bz):
        #eigen = np.vectorize(self.eigenvalues)
        egvals = self.eigenvalues(Bz)
        ms0 = (egvals[0] + egvals[1] + egvals[2])/3		# energy of 0 level averaged for simpler graph
        return np.array([egvals[3]-ms0, egvals[4]-ms0, egvals[5]-ms0, egvals[6]-ms0, egvals[7]-ms0, egvals[8]-ms0])

class ElectronicSpinHamiltonian(SpinHamiltonian):
    def __init__(self):
        SpinHamiltonian.__init__(self)
        self.ms0 = 0 # energy (frequency) of ms=0 state. Initialized to 0. It is later changed by the code.
        self.dpar = 0.03
        self.dperp = 0.17
        
    
    def interaction(self,E, theta, z):# z is a flag to distinguish between parallel and perpendicular
        Ex = E*math.cos(theta)		# calculating B from its magnitude, polar angle.
        Ey = E*math.sin(theta)
        Ez=0
        if z :
            Ex =0
            Ey =0
            Ez = E
            
        Hes = (self.dpar * (Ez*self.sz**2) -  
               self.dperp * (Ex*(self.sx**2-self.sy**2)) +  
               self.dpar * (Ey*(self.sx*self.sy + self.sy*self.sx)) )
        Hes = self.zero_field + Hes
        return Hes.eigenstates()

    def transition_freqs(self,E, theta, z):
        egvals = self.interaction(E,theta,z)[0]

        if(E == 0): self.ms0 = egvals[0] 
        f1 = egvals[2] - self.ms0 if(z) else egvals[2]-egvals[0] # to distinguish parallel and perpendiculr energies as qutip sorts them
        f0 = abs(egvals[1] + egvals[0] - (2*self.ms0)) if(z) else egvals[1]-egvals[0] # absolute value of frequency

        return np.array([f1,f0])

    def spin_projection(self,E, theta, z):
        egst = self.interaction(E, theta, z)

        msx0 = qt.expect(self.sx,egst[1][0])   # spin projection of 0th spin eigenstate of Hs on x axis
        msy0 = qt.expect(self.sy,egst[1][0])   # spin projection of 0th spin eigenstate of Hs on y axis
        msz0 = qt.expect(self.sz,egst[1][0])   # spin projection of 0th spin eigenstate of Hs on z axis

        msx1 = qt.expect(self.sx,egst[1][1])   # spin projection of 1st spin eigenstate of Hs on x axis
        msy1 = qt.expect(self.sy,egst[1][1])   # spin projection of 1st spin eigenstate of Hs on y axis
        msz1 = qt.expect(self.sz,egst[1][1])   # spin projection of 1st spin eigenstate of Hs on z axis

        msx2 = qt.expect(self.sx,egst[1][2])   # spin projection of 2nd spin eigenstate of Hs on x axis
        msy2 = qt.expect(self.sy,egst[1][2])   # spin projection of 2nd spin eigenstate of Hs on y axis
        msz2 = qt.expect(self.sz,egst[1][2])   # spin projection of 2nd spin eigenstate of Hs on z axis

        return np.array([msx0,msy0,msz0,msx1,msy1,msz1,msx2,msy2,msz2])

