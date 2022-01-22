import os
import re


def read_tsp(filename):
    """
    Read a file in .tsp format into a pandas DataFrame
    """
    filename_path = str(os.path.abspath(__file__)).replace("src/read_data.py", filename)
    with open(filename_path) as file:
        cities = []
        flag = False
        for line in file.readlines()[0:-1]:
            info_list = []
            if line.startswith('EOF'):
                break
            if line.startswith('NODE_COORD_SECTION'):
                flag = True
            elif flag == True:
                info = re.split('[ ]+', line.strip())
                info_list.append(int(info[0]))
                info_list.append(float(info[1]))
                info_list.append(float(info[2]))
                cities.append(info_list)
    return cities

def read_atsp_path(filename):
    return str(os.path.abspath(__file__)).replace("src/read_data.py", filename)

def inputTime():
    seconds = input("Please enter the maximum execution time in seconds: ")
    if len(seconds) == 0:
        print('Please input seconds')
        inputTime()
    else:
        if seconds.isdigit():
            return int(seconds)
        else:
            print('Please input seconds\n')
            inputTime()

def read_twtsp(filename):
    filename_path = str(os.path.abspath(__file__)).replace("src/read_data.py", filename)
    with open(filename_path) as file:
        cities = []
        for line in file.readlines()[0:-1]:
            info_list = []

            info = re.split('[ ]+', line.strip())
            if "CUST" in info:
                continue
            info_list.append(int(info[0]))
            info_list.append(float(info[1]))
            info_list.append(float(info[2]))
            info_list.append(float(info[3]))
            info_list.append(float(info[4]))
            info_list.append(float(info[5]))
            info_list.append(float(info[6]))
            cities.append(info_list)
    return cities