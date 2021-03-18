import numpy as np
import matplotlib.pyplot as plt

class UWHAM_FreeEnergy:
    def __init__(self,lower_range,upper_range,n_bins,\
                 well_width,well_height,well_type='double',\
                 friction_const=0.1,temperature=298,timestep_size=1e-4):
        '''
        Initialize the umbrella sampling/UWHAM/free energy class
        Inputs:
            self (class)
            lower_range = Lower bound on reaction coordinate (nm)
            upper_range = Upper bound on reaction coordinate (nm)
            n_bins = Number of bins to divide between lower and upper range
            well_width = Distance between potential energy wells (in nm)
            well_height = Distance between min and max energies (zJ)
            well_type = Type of well to be used (double or triple)
            friction_const = Frictional constant on Brownian particles
            temperature = Temperature of system in Kelvin
            timestep_size = MD simulation step size (in ps)
        Outputs:
            None
        '''
        self.kB = 1.38064852E-2 # zJ/K
        
        if well_type in ['double','triple']:
            self.well_type=well_type
        else:
            raise Exception(f'Invalid well type: {well_type}')
        
        self.lower=lower_range #nm
        self.upper=upper_range #nm
        self.n_bins=n_bins
        self.x_range = np.linspace(self.lower,self.upper,self.n_bins+1)
        
        self.well_width=well_width #nm
        self.well_height=well_height #zJ
        self.gamma = friction_const
        self.temp=temperature #Kelvin
        self.dt = timestep_size
        return
    
    def double_well_potential(self,x):
        '''
        Double well potential
        Inputs:
            self (class)
            x = Current position in reaction coordinate
        Outputs:
            Unbiased potential energy for the double well
        '''
        return (self.well_height/(self.well_width**4))*\
                    (x**2 - self.well_width**2)**2
    
    def double_well_force(self,x):
        '''
        Double well force
        Inputs:
            self (class)
            x = Current position in reaction coordinate
        Outputs:
            Unbiased force magnitude for the double well
        '''
        return (4*self.well_height*x/(self.well_width**4))*\
                (x**2 - self.well_width**2)
    
    def triple_well_potential(self,x):
        '''
        Triple well potential
        Inputs:
            self (class)
            x = Current position in reaction coordinate
        Outputs:
            Unbiased potential energy for the triple well
        '''
        return (27*self.well_height/(4 * self.well_width**6))*\
                (x**2 - self.well_width**2)**2 * (x**2)
    
    def triple_well_force(self,x):
        '''
        Triple well force
        Inputs:
            self (class)
            x = Current position in reaction coordinate
        Outputs:
            Unbiased force magnitude for the triple well
        '''
        return (27*self.well_height/(4 * self.well_width**6))*\
        (6*(x**5) - 8*(self.well_width**2 * x**3) + 2*(self.well_width**4 * x))
    
    def umbrella_harmonic_biasing_potential(self,x,x_bias,oscillator=None):
        '''
        Harmonic energy biasing to create the "umbrella" around some bias term
        Inputs:
            self (class)
            x = Current position in reaction coordinate (nm)
            x_bias = Left-hand-side biased position of reaction coordinate (nm)
            oscillator = effective "strength" of the energy biasing
        Outputs:
            Biased potential energy ascribed to the umbrella
        '''
        if oscillator is None:
            oscillator=50*self.kB*self.temp
        return oscillator*(x-x_bias)**2
    
    def umbrella_harmonic_biasing_force(self,x,x_bias,oscillator=None):
        '''
        Harmonic force biasing to create the "umbrella" around some bias term
        Inputs:
            self (class)
            x = Current position in reaction coordinate (nm)
            x_bias = Left-hand-side biased position of reaction coordinate (nm)
            oscillator = effective "strength" of the force biasing
        Outputs:
            Biased force magnitude ascribed to the umbrella
        '''
        if oscillator is None:
            oscillator=100*self.kB*self.temp
        return oscillator*(x-x_bias)
    
    def run_brownian_sim(self,timesteps,start_pos=None):
        '''
        Runs a Brownian simulation on a single particle with no biasing.
        Inputs:
            self (class)
            timesteps = number of iterations to perform for each simulation
            start_pos = starting position of the particle within lower/upper range
        Outputs:
            Times, trajectories, potential energies, and forces on Brownian particle
        '''
        self.timesteps=timesteps
        if start_pos is None:
            start_pos = np.random.uniform(-3,3,1)[0]
        
        if self.well_type=='double':
            PE_eq = self.double_well_potential
            F_eq = self.double_well_force
        else:
            PE_eq = self.triple_well_potential
            F_eq = self.triple_well_force
        
        trajectory_li = [start_pos]
        potential_energy_li = [PE_eq(start_pos)]
        force_mag_li = [F_eq(start_pos)]
        
        while len(trajectory_li)!=timesteps+1:
            next_step = trajectory_li[-1]-(((F_eq(trajectory_li[-1]))/self.gamma)+\
                        np.random.normal(0,2*self.kB*self.temp/self.gamma))*self.dt
            trajectory_li.append(next_step)
            potential_energy_li.append(PE_eq(next_step))
            force_mag_li.append(F_eq(next_step))
        
        self.all_times = np.linspace(0,timesteps,timesteps+1)*self.dt
        self.trajectory_li = np.array(trajectory_li)
        self.potential_energy_li = np.array(potential_energy_li)
        self.force_mag_li = np.array(force_mag_li)
        
        return self.all_times, self.trajectory_li, \
                self.potential_energy_li, self.force_mag_li
    
    
    def run_umbrella_sim_single(self,timesteps,x_bias,start_pos=None):
        '''
        Runs a full MD simulation across a single selected bin.
        Inputs:
            self (class)
            timesteps = number of iterations to perform for each simulation
            x_bias = left-hand-side starting bias for the harmonic oscillator
            start_pos = starting position of the particle within the bin range
        Outputs:
            Single biased particle times, trajectories, energies, and forces
        '''
        self.timesteps=timesteps
        if self.well_type=='double':
            PE_eq = self.double_well_potential
            F_eq = self.double_well_force
        else:
            PE_eq = self.triple_well_potential
            F_eq = self.triple_well_force
        
        if x_bias in self.x_range:
            x_bias_lower = x_bias
            x_bias_upper = self.x_range[self.x_range.tolist().index(x_bias)+1]
        else:
            raise Exception(f'{x_bias} is not a member of default self.x_range.')
        if start_pos is None:
            start_pos = np.random.uniform(x_bias_lower,x_bias_upper,1)[0]
        elif start_pos<x_bias_lower or start_pos>x_bias_upper: #Out of range
            raise Exception(f'{start_pos} invalid, select in [{x_bias_lower},{x_bias_upper}]')
        
        u_trajectory_li = [start_pos]
        u_potential_energy_li = [PE_eq(start_pos)]
        u_force_mag_li = [F_eq(start_pos)]
        
        while len(u_trajectory_li)!=timesteps+1:
            next_pos = u_trajectory_li[-1]-((F_eq(u_trajectory_li[-1])/self.gamma)+\
                            np.random.normal(0,(2*self.kB*self.temp/self.gamma))+\
                            self.umbrella_harmonic_biasing_force(u_trajectory_li[-1],\
                                                                 x_bias_lower))*self.dt
            u_trajectory_li.append(next_pos)
            u_potential_energy_li.append(PE_eq(next_pos))
            u_force_mag_li.append(F_eq(next_pos))
        
        self.u_all_times = np.linspace(0,timesteps,timesteps+1)*self.dt
        self.u_trajectory_li = np.array(u_trajectory_li)
        self.u_potential_energy_li = np.array(u_potential_energy_li)
        self.u_force_mag_li = np.array(u_force_mag_li)
        
        return self.u_all_times, self.u_trajectory_li, \
                self.u_potential_energy_li, self.u_force_mag_li
    
    def run_umbrella_sim_full(self,timesteps):
        '''
        Runs a full umbrella simulation across some x_range.
        Inputs:
            self (class)
            timesteps = number of iterations to perform for each simulation
        Outputs:
            All times, trajectories, potential energies and forces associated with each bin
        '''
        self.timesteps = timesteps
        all_u_traj = []
        all_u_pe = []
        all_u_force = []
        for x_bias in self.x_range[:-1]:
            _, trajs, pes, forces = self.run_umbrella_sim_single(timesteps,x_bias,start_pos=None)
            all_u_traj.append(trajs)
            all_u_pe.append(pes)
            all_u_force.append(forces)
        
        self.all_u_times = np.linspace(0,timesteps,timesteps+1)*self.dt
        self.all_u_traj = np.array(all_u_traj)
        self.all_u_pe = np.array(all_u_pe)
        self.all_u_force = np.array(all_u_force)
        return self.all_u_times, self.all_u_traj, self.all_u_pe, self.all_u_force
    
    def helmholtz_energy(self,lower_seek,upper_seek,maxiter=10000,threshold=0.1,print_modulus=None):
        '''
        This method computes the Helmholtz free energy after umbrella sampling has been performed.
        Shoutout to https://github.com/jjgoings/wham for being a strong source of inspiration.
        Inputs:
            self (class)
            lower_seek = starting position (in nm) of the reaction coordinate (set to 0 zJ for ref)
            upper_seek = final position (in nm) of the reaction coordinate
            max_iter = Maximum number of iterations to reach convergence
            threshold = Upper bound of allowed change between iterations to free energy convergence
            print_modulus = How you want to see updates on convergence (None means no prints)
        Outputs:
            Free energy changes across each bin of the simulation
        '''
        if lower_seek in self.x_range:
            lower_ind = self.x_range.tolist().index(lower_seek)
        else:
            raise Exception(f'Selected lower bound {lower_seek} not in x_range.')
        if upper_seek in self.x_range:
            upper_ind = self.x_range.tolist().index(upper_seek)
        else:
            raise Exception(f'Selected upper bound {upper_seek} not in x_range')
        
        if self.well_type=='double':
            energy_init = np.array([self.double_well_potential(i) \
                                    for i in self.x_range])[lower_ind:upper_ind+1]
            force_init = np.array([self.double_well_force(i) \
                                   for i in self.x_range])[lower_ind:upper_ind+1]
        else:
            energy_init = np.array([self.triple_well_potential(i) \
                                    for i in self.x_range])[lower_ind:upper_ind+1]
            force_init = np.array([self.triple_well_force(i) \
                                   for i in self.x_range])[lower_ind:upper_ind+1]
        bin_range = upper_ind-lower_ind
        
        energy_window = np.zeros((bin_range,self.timesteps+1,bin_range))
        force_window = np.ones(bin_range)
        mult_factor = (self.timesteps+1)*force_window
        
        force_window_update = np.zeros(bin_range)
        
        for i in range(bin_range):
            for j in range(bin_range):
                energy_window[i,:,j]=np.exp((self.kB*self.temp)*0.5*force_init[j]*\
                    (self.all_u_pe.T[:,i]-energy_init[j])**2)
        
        FE_conv_li = [np.log(force_window)/(self.kB*self.temp)]
        diff = np.inf
        counter=0
        while diff>threshold or counter<50: #Enforce min number of iters, here it is 50
            for i in range(bin_range):
                denominator = np.einsum('ilj,j->il',energy_window,mult_factor)**-1
                K_force_window = np.einsum('il,il',energy_window[:,:,i],denominator)
                force_window_update[i]=K_force_window
                force_window[i]=(force_window[0]*K_force_window)**-1
                mult_factor[i]=self.timesteps*force_window[i]
            diff=np.linalg.norm(np.log(force_window*force_window_update))
            force_window = force_window_update[0]/force_window_update
            counter+=1
            FE_conv_li.append(np.log(force_window)/(self.kB*self.temp))
            if print_modulus is not None:
                if len(FE_conv_li)%print_modulus == 0:
                    print(f'Difference to convergence (lower limit {threshold}): {diff}')
            if counter >=maxiter:
                raise Exception('Distribution of free energies failed to converge.')
        
        self.final_free_energy = np.log(force_window)/(self.kB*self.temp)
        return self.final_free_energy