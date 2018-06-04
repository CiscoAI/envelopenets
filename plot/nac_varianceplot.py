#Copyright 2018 Cisco Systems All Rights Reserved
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def remove_logging(line):
    line=re.sub("\d\d\d\d.*ops.cc:79\] ", "", line)
    return line

def get_cell_and_value(line):
    line = remove_logging(line)
    fields = line.split(':')
    if len(fields)  != 3:
       return None, None
    cell = fields[1]
    value = float(fields[2].strip().lstrip('[').rstrip(']'));
    return cell, value;

def filter_log(log):
    '''
       removes unnecessary lines from log file
    '''
    filter_string = 'Cell'
    metric="MeanSSS"
    filtered_log = [line for line in log if filter_string in line and metric in line]
    return filtered_log

def read_file(filename):
    with open(filename) as f:
        log = f.read()
    log = log.split('\n')
    return log

def process_data(log):
    stats = {}
    for line in log:
        cell, value = get_cell_and_value(line)
        if cell == None:
           continue;
        if cell not in stats:
            stats[cell] = []
        stats[cell].append(value)
    return stats

def get_data():
    file_name = './res.log'
    log = read_file(file_name)
    log = filter_log(log)
    stats = process_data(log)
    return stats

def plot_data(name, stats):
    fig = plt.figure()
    maxy = 0;
    miny = 100000
    for key in stats:
       for y  in stats[key]:
           if y  > maxy:
               maxy = y
           if y  < miny:
               miny = y
       plt.plot(stats[key], label=key)
    #ax = plt.axes()
    #ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    #plt.locator_params(axis='y', nticks=10)
    ytix = [x for x in range(int(0.8 * miny), int(1.2 * maxy))]
    plt.yticks(ytix)
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, -0.2) ,ncol=2)
    #plt.show()
    fig.savefig(name+".png", bbox_inches='tight');
    #fig.savefig(name+".png");

if __name__ == "__main__":
    stats = get_data()
    cellbranches = stats.keys();
    cells = []
    cellstats ={}
    #Input data is indexed by cell-branch
    #Index by cell
    for cellbranch in cellbranches:
        [cell, branch] = cellbranch.split("/");
        if cell not in cellstats:
            cellstats[cell] = {}
        cellstats[cell][cellbranch] = stats[cellbranch]
 
    for cell in cellstats: 
        plot_data(cell, cellstats[cell]);
