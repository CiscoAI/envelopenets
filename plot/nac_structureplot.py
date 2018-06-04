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
import json
import numpy as np

def process():
    files = []
    archnames = ["bc.trial-1", "bc.trial-2", "bc.trial-3", "bc.trial-4"] #, "bc.trial-5", "bc.trial-6", "bc.trial-7", "bc.trial-8", "bc.trial-9", "bc.trial-10"]
    for archname in archnames:
        f = "network."+archname+".json"
        files.append(f)

    widths = {}
    for f in files:
        with open(f, 'r') as archfile:
            arch = json.load(archfile);
            #arch = data['arch']
            print arch
            lidx = 0;
            maxwidth = 0;
            for layer in arch:
                print layer
                if 'filters' in layer:
                    width  = len(layer["filters"].keys())
                    if width > maxwidth:
                        maxwidth = width
                    if lidx not in widths:
                            widths[lidx] = [];
                    widths[lidx].append(width);
                    lidx+=1

    widthmean =[] 
    widthstddev = [] 
    for layer in widths:
        widthmean.append(np.mean(widths[layer]))
        widthstddev.append(np.std(widths[layer]))
    print widths
    print widthmean
    print widthstddev

    plot_shapedist(widthmean, widthstddev, maxwidth);

def plot_shapedist(mean, stddev, maxy):
    fig = plt.figure()
    color_mean = 'g--'
    color_shading = 'g'
    ub = np.add(mean , stddev)
    lb = np.add(mean , np.multiply(-1, stddev))
    plt.fill_between(range(len(mean)), ub, lb, color=color_shading, alpha=.5)
    plt.plot(mean, color_mean)

    plt.xlabel("Depth")
    plt.ylabel("Average width")
    ytix = [x for x in range(0, maxy+1)]
    plt.yticks(ytix)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)
    fig.savefig("structure.png", bbox_inches='tight');

if __name__ == "__main__":
    process()
