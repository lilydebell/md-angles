# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import math

import MDAnalysis 

from datetime import datetime


# -

class Orientation:
    '''Find the dipole orientation and birefringence from a trajectory file.'''
    global u
    
    def __init__(self):
        self.dipole_vectors = np.empty(3)
        
    def find_dipole_vector(self, position_file, trajectory_file):
        '''This function finds the dipole vector of each water molecule.'''
        
        def create_universe(self, position_file, trajectory_file):
            '''This function creates a universe object.'''
            u = MDAnalysis.Universe(position_filename, trajectory_filename, format='TRR', topology_format='GRO')
            return u
            
        def read_OW_coordinates(u):
            '''This function reads the coordinates of oxygen atoms in water molecules and returns an array of all coordinates.'''
            OW_select = u.select_atoms('name OW')
            OW_list = []
            for ts in u.trajectory:
                OW_list.append(OW_select.positions)
            OW_array = np.array(OW_list)
            return OW_array

        def read_HW1_coordinates(u):
            '''This function reads the coordinates of one set of hydrogren atoms in water molecules and returns an array of all coordinates.'''
            HW1_select = u.select_atoms('name HW1')
            HW1_list = []
            for ts in u.trajectory:
                HW1_list.append(HW1_select.positions)
            HW1_array = np.array(HW1_list)
            return HW1_array

        def read_HW2_coordinates(u):
            '''This function reads the coordinates of one set of hydrogren atoms in water molecules and returns an array of all coordinates.'''
            HW2_select = u.select_atoms('name HW2')
            HW2_list = []
            for ts in u.trajectory:
                HW2_list.append(HW2_select.positions)
            HW2_array = np.array(HW2_list)
            return HW2_array
        
        vector_1_array = read_HW1_coordinates(create_universe(self, position_file, trajectory_file)) - read_OW_coordinates(create_universe(self, position_file, trajectory_file))
        vector_2_array = read_HW2_coordinates(create_universe(self, position_file, trajectory_file)) - read_OW_coordinates(create_universe(self, position_file, trajectory_file))
        self.dipole_vectors = vector_1_array + vector_2_array
        return self.dipole_vectors
    
    def angle_calc(self):
        ''' This function calculates the cosine of the angle formed by the y axis and dipole vector of each water molecule.'''
        # intialize variables
        ref_vector_x = np.array([1.000, 0.000, 0.000])
        ref_vector_y = np.array([0.000, 1.000, 0.000])
        ref_vector_z = np.array([0.000, 0.000, 1.000])
        unit_vector_x = ref_vector_x / np.linalg.norm(ref_vector_x)
        unit_vector_y = ref_vector_y / np.linalg.norm(ref_vector_y)
        unit_vector_z = ref_vector_z / np.linalg.norm(ref_vector_z)
        time_list = []
        # normalize vector array
        norm_array = np.linalg.norm(self.dipole_vectors, axis = 2)
        norm_array_final = norm_array[:,:, np.newaxis]
        unit_vector_array = self.dipole_vectors / norm_array_final
        #take the dot product of every element of the array with the a unit vector
        x_data = np.dot(unit_vector_array, unit_vector_x)
        x_squared_data = (np.dot(unit_vector_array, unit_vector_x))**2
        y_data = np.dot(unit_vector_array, unit_vector_y)
        y_squared_data = (np.dot(unit_vector_array, unit_vector_y))**2
        z_data = np.dot(unit_vector_array, unit_vector_z)
        z_squared_data = (np.dot(unit_vector_array, unit_vector_z))**2
        # build a dataframe for cos(theta) and cos^2(theta) with the dot product array
        df_x = pd.DataFrame(data = x_data)
        df_y = pd.DataFrame(data = y_data)
        df_z = pd.DataFrame(data = z_data)
        df_squared_x = pd.DataFrame(data = x_squared_data)
        df_squared_y = pd.DataFrame(data = y_squared_data)
        df_squared_z = pd.DataFrame(data = z_squared_data)
        average_x = df_x.mean(axis = 1)
        average_y = df_y.mean(axis = 1)
        average_z = df_z.mean(axis = 1)
        average_squared_x = df_squared_x.mean(axis = 1)
        average_squared_y = df_squared_y.mean(axis = 1)
        average_squared_z = df_squared_z.mean(axis = 1)
        average_df_x = pd.DataFrame(data = average_x, columns = ['cos(theta) (x)'])
        average_df_y = pd.DataFrame(data = average_y, columns = ['cos(theta) (y)'])
        average_df_z = pd.DataFrame(data = average_z, columns = ['cos(theta) (z)'])
        average_squared_df_x = pd.DataFrame(data = average_squared_x, columns = ['cos^2(theta) (x)'])
        average_squared_df_y = pd.DataFrame(data = average_squared_y, columns = ['cos^2(theta) (y)'])
        average_squared_df_z = pd.DataFrame(data = average_squared_z, columns = ['cos^2(theta) (z)'])
        frames = [average_df_x, average_df_y, average_df_z, average_squared_df_x, average_squared_df_y, average_squared_df_z]
        all_data = pd.concat(frames, axis = 1)
        for i in range(0, len(average_x)):
            time_list.append(i*2)
        all_data['time (fs)'] = time_list
        self.cos_dataframe = all_data
        return self.cos_dataframe
    
    def find_birefringence(self):
        '''This function finds the birefringence from the cos^2(theta) data previously calculated.'''
        # initialize variables
        time_list = []
        alpha_parallel = 1.495
        alpha_perpendicular = 0.5 * (1.626 + 1.286)
        alpha_hat_G = (1/3) * (1.626 + 1.286 + 1.495)
        n_naught = math.sqrt(1 + 4 * math.pi * (1580/(2.3**3))) * alpha_hat_G
        # convert dataframe passed as argument to an array
        x_array = self.cos_dataframe['cos^2(theta) (x)'].to_numpy()
        y_array = self.cos_dataframe['cos^2(theta) (y)'].to_numpy()
        z_array = self.cos_dataframe['cos^2(theta) (z)'].to_numpy()
        # find the bifringence
        x_birefringence = ((4 * math.pi * 1580) / (2.3**3 * n_naught)) * (alpha_parallel - alpha_perpendicular) * (x_array - (1/3)) 
        y_birefringence = ((4 * math.pi * 1580) / (2.3**3 * n_naught)) * (alpha_parallel - alpha_perpendicular) * (y_array - (1/3))
        z_birefringence = ((4 * math.pi * 1580) / (2.3**3 * n_naught)) * (alpha_parallel - alpha_perpendicular) * (z_array - (1/3))
        # convert array to dataframe
        df_x = pd.DataFrame(data = x_birefringence, columns = ['n(t) (x)'])
        df_y = pd.DataFrame(data = y_birefringence, columns = ['n(t) (y)'])
        df_z = pd.DataFrame(data = z_birefringence, columns = ['n(t) (z)'])
        frames = [df_x, df_y, df_z]
        all_data = pd.concat(frames, axis = 1)
        for i in range(0, len(df_x)):
            time_list.append(i*2)
        all_data['time (fs)'] = time_list
        self.birefringence_dataframe = all_data
        return self.birefringence_dataframe
    
    def dataframe_to_csv(self):
        '''This function converts dataframes to CSVs and saves them in the home directory.'''
        self.cos_dataframe.to_csv('cos_theta' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.birefringence_dataframe.to_csv('birefringence' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        


if __name__ == '__main__':
    o = Orientation()
    while True:
        position_filename = input('Enter the position file name: ')
        trajectory_filename = input('Enter the trajectory file name: ')
        o.find_dipole_vector(position_filename, trajectory_filename)
        o.angle_calc()
        o.find_birefringence()
        o.dataframe_to_csv()
        proceed = input('Would you like to continue (y/n)?')
        if proceed == 'n':
            break


