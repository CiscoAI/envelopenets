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

import os
import json

def single_trial(configfile):
   #Principle: Construct
   cmd = "python nac_gen.py "+configfile
   os.system(cmd)

def multiple_trials(configfile, trials):
    for trial_num in range(1, trials):
        with open(configfile, 'r+') as f:
            data = json.load(f)
            data['parameters']['archname'] += '-trial-' + str(trial_num)
        nconfigfile = configfile+"-"+str(trial_num)
        with open(nconfigfile, 'w') as f2:
            json.dump(data, f2, indent=4)
        cmd = "python nac_gen.py "+nconfigfile
        os.system(cmd)

if __name__ == "__main__":
    #Principle: Construct: Best case/Worst case/Rnd
    #configs=['cons-exp-3.bc', 'cons-exp-3.wc', 'cons-exp-3-rnd']
    configs=['cons-exp-3.bc']
    for config in configs:
         configfile = './configs/v2.0/construction/network.'+config+'.json'
         single_trial(configfile)


    #Copy the generated arch from the construction run to the long run  config files before running these
    #Long runs (Best case/Worst case)
    #configs=['cons-exp-3-bc-longrun', 'cons-exp-3-wc-longrun']
    configs=['cons-exp-3-bc-longrun']
    for config in configs:
         configfile = './configs/v2.0/longruns/network.'+config+'.json'
         single_trial(configfile)
    

    #Long runs (Rnd nets)
    #configs = ['cons-exp-3-rnd1-longrun', 'cons-exp-3-rnd2-longrun', 'cons-exp-3-rnd3-longrun', 'cons-exp-3-rnd4-longrun', 'cons-exp-3-rnd5-longrun', 'cons-exp-3-rnd6-longrun', 'cons-exp-3-rnd7-longrun', 'cons-exp-3-rnd8-longrun', 'cons-exp-3-rnd9-longrun', 'cons-exp-3-rnd10-longrun']
    configs = ['cons-exp-3-rnd1-longrun']
    for config in configs:
         configfile = './configs/v2.0/longruns/network.'+config+'.json'
         single_trial(configfile)

    #Structural analysis
    configs = ['cons-exp-3.bc']
    numtrials = 6
    for config in configs:
         configfile = './configs/v2.0/construction/network.'+config+'.json'
         multiple_trials(configfile, numtrials)
