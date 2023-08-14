import os
import sys
import glob
import netCDF4
import multiprocessing
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

def check_variable_exists(PLUMBER2_path, site_name, model_names, key_word, key_word_not="None"):

    # file path

    my_dict      = {} 
    for j, model_name in enumerate(model_names):
        print(model_name)

        # Set input file path
        file_path    = glob.glob(PLUMBER2_path+model_name +"/*"+site_name+"*.nc")
        var_exist    = False

        try:
            with netCDF4.Dataset(file_path[0], 'r') as dataset:
                for var_name in dataset.variables:
                    print(var_name)
                    variable  = dataset.variables[var_name]
                    long_name = variable.long_name.lower()  # Convert description to lowercase for case-insensitive search

                    if key_word in long_name and key_word_not not in long_name:
                        print(long_name)
                        my_dict[model_name] = var_name
                        var_exist = True
                        print(f"The word '{key_word}' is in the description of variable '{var_name}'.")
                        break  # Exit the loop once a variable is found

        except Exception as e:
            print(f"An error occurred: {e}")

        # variable doesn't exist
        if not var_exist:
            my_dict[model_name] = 'None'

    # f = open(f"./txt/{site_name}_{key_word}.txt", "w")
    # f.write(str(my_dict))
    # f.close()
    return my_dict


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"

    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    # The name of models
    model_names        = [ "CABLE","3km27","CABLE-POP-CN","CHTESSEL_Ref_exp1",
                           "GFDL","MATSIRO","NASAEnt","ORCHIDEE_tag2.1",
                           "QUINCY","ACASA","CHTESSEL_ERA5_3","CLM5a",
                           "JULES_GL9","LSTM_raw","MuSICA","NoahMPv401","ORCHIDEE_tag3_2",
                           "RF","STEMMUS-SCOPE","LPJ-GUESS","SDGVM", "BEPS"]

    # Read all site names
    all_site_path      = sorted(glob.glob(PLUMBER2_met_path+"/*.nc"))
    # site_names         = [os.path.basename(site_path).split("_")[0] for site_path in all_site_path]
    site_names = ['AU-Cow']
    # Key word for searching 
    key_word           = "trans"
    key_word_not       = "evap"
    for site_name in site_names:
        print(site_name)
        check_variable_exists(PLUMBER2_path, site_name, model_names, key_word, key_word_not)



# if __name__ == "__main__":
#     # Input argument for tasks
#     input_arg = 5

#     # Create two processes for the tasks
#     process1 = multiprocessing.Process(target=task1, args=(input_arg,))
#     process2 = multiprocessing.Process(target=task2, args=(input_arg,))

#     # Start both processes
#     process1.start()
#     process2.start()

#     # Wait for both processes to complete
#     process1.join()
#     process2.join()

#     # Retrieve return values from the processes
#     result1 = process1.exitcode  # Replace with the actual way to get return value
#     result2 = process2.exitcode  # Replace with the actual way to get return value

#     print("Both tasks have completed.")