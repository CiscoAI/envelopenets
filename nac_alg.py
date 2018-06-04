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
import itertools
import datetime
import json
import copy
import re
import random
import math

'''
NAC Algorithm: 
- Generation using EnvelopNets
- Random network generation
'''
class NACAlg:
    def __init__(self, algorithm, config):
        now = datetime.datetime.now()
        self.ts =now.strftime("%m-%d-%Y")
        self.algorithm =algorithm
        self.config = config
        print(self.config);
        if self.algorithm == "envelopenet" or self.algorithm == "envelopenet2":
            self.maxfilterprune = int(self.config["maxfilterprune"])
            if 'worstcase' in self.config:
                self.worstcase = self.config["worstcase"]
            else:
                self.worstcase = False
            self.envelopecell = self.config["envelopecell"]
            self.initcell = self.config["initcell"]
            self.layers = self.config["layersperstage"]
            self.maxlayers = self.config["maxlayersperstage"]
            self.stages = int(self.config["stages"])
            self.parameterlimits = self.config["parameterlimits"]
            self.construction = self.config["construction"]
        elif self.algorithm == "random":
            self.parameterlimits = self.config["parameterlimits"]
            self.layers = self.config["layersperstage"]
            self.stages = int(self.config["stages"])
            self.numblocks = self.config["numblocksperstage"]
            self.blocks = (self.config["blocks"])
        elif self.algorithm == "deterministic":
            self.arch = self.config["arch"]
        else:
           print ("Invalid algorithm")
           exit(-1); 
    
    def __del__(self):
        pass
    
    def generate(self):
        '''
        Generate a network arch based on network config params: Used for 
        "oneshot" mode or the initial network when run in "construct" mode
        '''
        if self.algorithm == "deterministic":
            return self.config["arch"]
        elif self.algorithm == "envelopenet":
            return self.gen_envelopenet_bystages()
        elif self.algorithm == "random":
            return self.gen_randomnet()
        else:
           print ("Invalid algorithm")
           exit(-1); 

    def construct(self, arch, samples):
        '''
        Construct a new network based on current arch, metrics from last run and 
        construction config params
        '''
        self.arch = arch
        if self.algorithm == "envelopenet":
            return self.construct_envelopenet_bystages(samples)
        elif self.algorithm == "random":
            return self.construct_random()
        else:
           print ("Invalid algorithm")
           exit(-1); 

    def construct_random(self):
        #Return a random network, equivalent to a network create 
        #by an envelope net generation algorithm, but with random pruning
        narch = self.gen_randomnet()
        narch = self.insert_skip(narch)
        return narch

    def construct_envelopenet_bystages(self, samples):
         with open(samples) as f:
            self.samples = f.read()
            self.samples = self.samples.split('\n')
         worstcase = self.worstcase
         stages = []
         stage = []
         stagecellnames = {};
         ssidx = {};
         #Cell0 is the init block
         lidx=1;
         ssidx[0] = 1;
         stagenum =0;
         for layer in self.arch:
             if 'widener' in layer:
                 lidx+=1
                 stages.append(stage)
                 stage = []
                 stagenum+=1
                 ssidx[stagenum] =lidx; 
             else:
                 for branch in layer["filters"]:
                     #TODO: Add the cellname to the config file the same as cell names in the logfiles
                     cellname = 'Cell'+str(lidx)+"/"+branch; 
                     if stagenum not in stagecellnames:
                         stagecellnames[stagenum] = []
                     stagecellnames[stagenum].append(cellname)
                 stage.append(layer);
                 lidx+=1
         stages.append(stage)
         stagenum=0
         narch =[]
         print ("Stage cellnames: "+str(stagecellnames));
         print ("Stage ssidx: "+str(ssidx));
         print ("Stages: "+str(stages));
         for stage in stages:
             if self.construction[stagenum] == True and len(stage) <= self.maxlayers[stagenum]:
                 prune = self.select_prunable(stagecellnames[stagenum], worstcase = worstcase);
                 print ("Stage: "+str(stage))
                 print ("Pruning "+str(prune))
                 nstage = self.prune_filters(ssidx[stagenum], stage, prune)
                 nstage = self.add_cell(nstage)
             else:
                 nstage = copy.deepcopy(stage);
             #Do not add widener for the last stage
             self.set_outputs(nstage, stagenum)
             if stagenum != len(stages) - 1:
                 nstage = self.add_widener(nstage)
             print("New stage :"+str(nstage))
             narch+=(nstage);
             stagenum+=1

         self.insert_skip(narch)
         print("Old arch :" +str( self.arch))
         print("New arch :" +str( narch))
         return narch     

    def select_prunablerandom(self, arch, number):
        allcells = [];
        cells = []
        return cells;  
 

    def remove_logging(self, line):
         line=re.sub("\d\d\d\d.*ops.cc:79\] ", "", line)
         return line

    def filter_samples(self, samples):
        #filter_string = 'Variance'
        filter_string = 'MeanSSS'
        filtered_log = [line for line in samples if filter_string in line]
        return filtered_log

    def get_filtersample(self, sample):
        fields = sample.split(":")
        metric = fields[0]
        filt = fields[1]
        value = float(fields[2].split(']')[0].lstrip('['));
        return filt, value
    
    def set_outputs(self, stage, stagenum):
        init = self.config["initcell"];
        print (init)
        sinit = sorted(init.keys()); #, key=init.get); #, reverse = True)
        #Input channels = output of last layer (conv) in the init 
        for layer in sinit:
            print(sinit)
            for branch in init[layer]:
                if "outputs" in init[layer][branch]:
                    inputchannels = init[layer][branch]["outputs"];
        print ("Input channels: "+str(inputchannels)); 
        W = math.pow(2, stagenum) * inputchannels
        print ("W : "+str(W))
        if self.parameterlimits[stagenum] == True:
            '''
            Parameter limiting: Calculate output of the internal filters such that 
            overall  params is maintained constant, see paper.
            '''
            layers = float(len(stage))
            print ("N : "+str(layers))
            outputs = int ((W / (layers-2.0)) * ( math.pow( layers- 1.0, 0.5) - 1))
            print ("outputs : "+str(outputs))
        print (stage)
        lidx=0
        for layer in stage:
            print (lidx)
            print (len(stage) - 1)
            if "widener" in layer:
                 print ("Widener");
                 lidx+=1
                 continue
            if lidx == len(stage) -1 or self.parameterlimits[stagenum] == False:
                 print ("Setting outputs to W");
                 layer["outputs"] = int(W)
            elif "filters" in layer:
                 print ("Limiting outputs");
                 layer["outputs"] = outputs
            lidx+=1
 
    def select_prunable(self, stagecellnames, worstcase = False):
        samples = self.filter_samples(self.samples)
        print (samples)
        measurements ={}
        for sample in samples:
             if sample == '':
                continue;
             sample = self.remove_logging(sample)
             filt, value = self.get_filtersample(sample)

             #Prune only filters in this stage
             if filt not in stagecellnames:
                 continue

             if filt not in measurements:
                  measurements[filt] = []
             measurements[filt].append(value)

        print ("Stage cell names"+str(stagecellnames))
        print ("Filter in samples "+str(list(measurements.keys())))
        #Rank variances, select filters to prune
        #Use last variance reading
        variances = {};
        for filt in measurements:
            variances[filt] = measurements[filt][-1];

        if worstcase == True:
            print("WARNING: -------GENERATING WORST CASE NET---------")
            reverse = True
        else:
            reverse = False
        svariances = sorted(variances, key=variances.get, reverse = reverse)
        #Count number of cells in each layer
        print ("All variances: "+ str(variances))
        print ("Sorted variances: "+ str(svariances))
        cellcount = {} 
        for cellbr in variances:
                cellidx = cellbr.split("/")[0].lstrip("Cell")
                if cellidx not in cellcount:
                    cellcount[cellidx]=0
                cellcount[cellidx]+=1

        totalfilterlayers = cellcount.keys();
        print (cellcount)
        #Make sure we do not prune all cells in one layer
        prunedcount ={}
        prune =[]
        for s in svariances:
             prunecellidx = s.split("/")[0].lstrip("Cell");
             if prunecellidx not in prunedcount:
                   prunedcount[prunecellidx] = 0;
             if prunedcount[prunecellidx]+ 1 < cellcount[prunecellidx]:
                 print("Pruning "+s)
                 prune.append(s)
                 prunedcount[prunecellidx]+=1
                 #Limit number of pruned cells to min of threshold * number of filters in stage and maxfilter prune
                 #TODO: Move thresold to config, make configurable per stage
                 #If the threshold is high enough and there are few filters in stage, only one will be pruned 
                 threshold = (1.0/3.0);
                 prunecount = min (self.maxfilterprune, int(threshold * float(len(stagecellnames))))
                 if len(prune) >=prunecount:
                      break;
        if len(prune) == 0:
             print (svariances)
             print ("Error: No cells to prune");
             exit(-1);
        return prune

    def prune_filters(self,ssidx, stage, prune):
        print(("Pruning "+str(prune)))
        #Generate a  pruned network without the wideners
        narch = []; 
        #= copy.deepcopy(self.arch);
        lidx = 0 
        nfilterlayers=0
        #for layer in self.arch:
        for layer in stage:
           if 'widener' in layer:
               lidx+=1
               continue;
           print("Layer "+str(lidx))
           print("Arch "+str(layer))
           #narch.append(copy.deepcopy(self.arch[lidx]));
           narch.append(copy.deepcopy(stage[lidx]));
           #for filt in self.arch[lidx]["filters"]:
           for filt in stage[lidx]["filters"]:
               fidx = int(filt.lstrip("Branch"));
               for p in prune:
                   print ("Checking "+str(p)+ " with :"+str(ssidx +lidx) + ":"+str(filt))
                   prunecidx = p.split("/")[0].lstrip("Cell");
                   prunefidx = p.split("/")[1].lstrip("Branch");
                   if ssidx + lidx == (int(prunecidx)) and fidx == int(prunefidx):
                       print ("Match")
                       del narch[-1]["filters"]["Branch"+str(prunefidx)];
           print ("Narc: "+str(narch[-1]))
           nfilterlayers+=1;
           lidx+=1
        return narch


    def add_cell(self, narch):
        narch.append({ "filters": self.envelopecell});
        # {"Branch0": "3x3", "Branch1": "3x3sep", "Branch2": "5x5", "Branch3": "5x5sep"} })
        return narch
    
    def add_widener(self, narch):
        narch.append({ "widener": {}});
        # {"Branch0": "3x3", "Branch1": "3x3sep", "Branch2": "5x5", "Branch3": "5x5sep"} })
        return narch

    def insert_skip(self, narch):
        if "skip" not in self.config:
            return narch
        if self.config["skip"] != True:
            return narch
        print(narch)
        for layer in narch:
            if "filters" in layer:
                 layer["inputs"] = "all"
        return narch 
    def insert_wideners(self, narch):
        #Insert wideners, 
        #Space maxwideners equally with a minimum spacing of self.minwidenerintval
        #Last widenerintval may have less layers than others
         
        #widenerintval= nfilterlayers//self.maxwideners
        widenerintval= len(narch)//self.maxwideners
        if widenerintval < self.minwidenerintval:
            widenerintval = self.minwidenerintval
        print ("Widener interval = "+str(widenerintval))
        nlayer=1
        insertindices=[];
        for layer in narch:
            print(str(nlayer))
            #Do not add a widener if it is the last layer
            if nlayer % widenerintval == 0 and nlayer != len(narch):
                insertindices.append(nlayer);
            nlayer+=1
        print ("Inserting wideners: "+str(insertindices));
        idxcnt = 0;
        for layeridx in insertindices:
                lidx = layeridx+idxcnt
                #Adjust insertion indices after inserts
                print("Adding widener"+ str(lidx))
                narch.insert(lidx , {"widener": {}})
                idxcnt+=1
        for layer in narch:
            print(layer)
        return narch
    
    def gen_randomnet(self):
        self.arch=[];
        for stage in range(self.stages):
            starch = []
            for idx in range(int(self.layers[stage])):
                starch.append({"filters":{} });
            self.set_outputs(starch, stage,);
            self.arch+=starch
            if stage != self.stages - 1:
                self.arch = self.add_widener(self.arch);
        print (self.arch)
        layer = 0;
        for stage in range(self.stages):
            #First add at least one block to each layer, to make sure that no layer has zero blocks
            for slayer in range(0, self.layers[stage]):
                block = random.randint(0, len(self.blocks) - 1)
                blockname = self.blocks[block];
                self.arch[layer]["filters"]["Branch0"] = blockname 
                layer+=1
            #Widener
            layer+=1
        print (self.arch)
   
        startlayer = 0; 
        for stage in range(self.stages):
            for idx in range(0, self.numblocks[stage] - self.layers[stage]):
                #Pick a random layer
                rlayer = random.randint(0, self.layers[stage] - 1)
                #Pick a random block
                block = random.randint(0, len(self.blocks) - 1)
                blockname = self.blocks[block];
                #Increment branch
                alayer = startlayer + rlayer
                print("Start, r, a"+str(startlayer)+":"+str(rlayer)+":"+str(layer));
                branch = len(self.arch[alayer]["filters"].keys())
                branchname = "Branch"+str(branch)
                self.arch[alayer]["filters"][branchname] = blockname 
            #Widener
            startlayer+=(self.layers[stage]+1)
        self.arch = self.insert_skip(self.arch)
        print (json.dumps( self.arch, indent=4, sort_keys=True))
        return self.arch

    def gen_envelopenet_bystages(self):
        self.arch=[];
        print ("Stages: "+str(self.stages))
        print ("Layerperstage: "+str(self.layers))
        for stageidx in range(int(self.stages)):
            print ("Stage: "+str(stageidx))
            stage = []
            for idx1 in range(int(self.layers[stageidx])):
                 print("Layer : "+str(idx1))
                 #TODO  Move this to an evelopenet gen function
                 #TODO: Add skip connections
                 stage.append({"filters": self.envelopecell});
            self.set_outputs(stage, stageidx)
            if stageidx != int(self.stages) - 1:
                 stage = self.add_widener(stage)
            self.arch+=stage;
        self.insert_skip(self.arch)
        print(json.dumps(self.arch, indent=4, sort_keys=True))
        return self.arch
