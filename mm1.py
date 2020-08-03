#mm1.py
"""Simulation of M/M/1 FIFO queue"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

#----------------------------------------------------------------
#INPUT

while True:
	try:
		sim_time = float(input("Insert simulation time, time units (tu): "))
		AT_rate = float(input("Insert arrival rate, lambda (customers/tu): "))
		ST_rate = float(input("Insert service rate, mu (customers/tu): "))

		if math.floor(sim_time) | math.floor(AT_rate) | math.floor(ST_rate) < 0.0:
			raise ValueError

	except ValueError:
		print('Values entered must be positive numbers. Please, try again:')
	else:
		break		

rho = AT_rate/ST_rate 

#----------------------------------------------------------------
#FUNCTION DEFINITIONS								

def simulation_mm1(AT_rate, ST_rate, sim_time):

	t = 0					#discrete time reference updated whenever consecutive events happen
	departure_t = 0
	arrival_t = 0

	clock = np.array([0])	#stores all event timestamps
	n_queue = np.array([0])	#stores queue length, updated everytime an event occurs (arrival/departure)
	queue = np.array([])	#stores arrival times of customers
	ctimes = np.array([])	#stores waiting times for clients, departure - arrival
		 
	#initialize simulation
	IAT = np.random.exponential(1/AT_rate) 						
	arrival_t += IAT 							#generate interarrival time and add to arrival times
	t = arrival_t		

	queue = np.append(queue, arrival_t)			#add client to queue
	clock = np.append(clock, t)
	n_queue = np.append(n_queue, len(queue))		

	ST = np.random.exponential(1/ST_rate)							
	departure_t = arrival_t + ST 				#generate departure time
	ctimes = np.append(ctimes, departure_t - queue[0]) 

	while not t > sim_time:

		IAT = np.random.exponential(1/AT_rate)					
		arrival_t += IAT
		queue = np.append(queue, arrival_t)						

		while arrival_t >= departure_t:

			t = departure_t
			clock = np.append(clock, t)
			queue = np.flip(queue[-1:0:-1])		#customer leaves the system		
							
			ST = np.random.exponential(1/ST_rate)

			#departure time for the next customer

			if queue[0] < departure_t:			#system not empty upon arrival departure time starts from previous departure
				departure_t += ST	
			else:								#system empty upon arrival departure time starts from arrival time
				departure_t = queue[0] + ST
					
			n_queue = np.append(n_queue, len(queue) - 1) 	#subtract len queue by 1 since last arrival is after time t
			ctimes = np.append(ctimes, departure_t - queue[0])
									
		t = arrival_t
		clock = np.append(clock, t)
		n_queue = np.append(n_queue, len(queue))		
															
	return n_queue, ctimes, clock

def sim_plots(n_queue, ctimes, clock):

	fig, (ax1, ax2) = plt.subplots(ncols= 2, nrows=1)
	fig.subplots_adjust(wspace = 0.3)

	sns.set_style('whitegrid')

	ax1 = sns.lineplot(clock, n_queue, drawstyle='steps-post', ax = ax1)	#queue size step plot
	ax1.set_title('N customers in system')
	ax1.set(xlabel='time (tu)', ylabel='n clients')

	ax2 = sns.lineplot(np.arange(len(ctimes)), ctimes, ax = ax2)			#waiting time plot 
	ax2.set_title('Waiting time customers')
	ax2.set(xlabel='client index', ylabel='waiting time (tu)')

	ax1.legend(title = f'Mean nclients {round(np.mean(n_queue),3)}')
	ax2.legend(title = f'Mean wait time :{round(np.mean(ctimes),3)}')
	plt.show()

#----------------------------------------------------------------
#PROGRAM EXECUTION

n_queue, ctimes, clock = simulation_mm1(AT_rate, ST_rate, sim_time)
sim_plots(n_queue, ctimes, clock)

