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
import csv



def get_data():

    fig = plt.figure()
    files = []
    labels = ['NAC', 'EnvelopeNet', 'Worst case (construct by \n pruning the best filters)']
    rndarchnames = ['rnd1','rnd2', 'rnd3', 'rnd4', 'rnd5', 'rnd6', 'rnd7', 'rnd8', 'rnd9', 'rnd10']
    algarchnames = ["bc", "en", "wc"]
    markers = ["s", "o", '^']
    for archname in algarchnames:
        f = "run_"+archname+"-tag-Precision @ 1.csv"
        files.append(f)
    fig = plt.figure()
    keyidx=0
    for f in files:
        x =[]
        y=[]
        with open(f, 'r') as csvfile:
            label = labels[keyidx]
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)
            for row in reader:
                x.append(float(row[1]));
                y.append(float(row[2]));
            plt.plot(x, y, label=label, marker=markers[keyidx])
        keyidx+=1;

    files = []
    for archname in rndarchnames:
        f = "run_"+archname+"-tag-Precision @ 1.csv"
        files.append(f)
    rx= {}
    ry = {}
    for f in files:
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)
            idx=0
            for row in reader:
                if idx not in rx:
                   rx[idx] = []
                   ry[idx] = []
                rx[idx].append(float(row[1]));
                ry[idx].append(float(row[2]));
                idx+=1
    rndmean = []
    rndstddev = []
    xaxis = []
    for idx in rx:
         xaxis.append(rx[idx][0])
         rndmean.append(np.mean(ry[idx]))
         rndstddev.append(np.std(ry[idx]))

    plt.errorbar(xaxis, rndmean, yerr=rndstddev, label="Average of 10 randomly \n generated networks")

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")

    ytix = [x/100.0 for x in range(85, 97)]
    plt.yticks(ytix)
    #xtix = [x/10.0 for x in range(5, 11)]
    #plt.xticks(xtix)
    #plt.legend(loc='lower center', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.45),
          fancybox=True, shadow=True, ncol=1)
    #plt.show()
    fig.savefig("performance.png", bbox_inches='tight');
    #fig.savefig(name+".png");

if __name__ == "__main__":
    stats = get_data()
