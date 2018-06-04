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
import sys


import nac_alg

"""
NAC generation
"""
class NAC:
    def __init__(self):
        self.archs = []
        now = datetime.datetime.now()
        self.basedir = "."
        self.datadir = self.basedir + "/data/"
        self.codedir=self.basedir
        self.count_params = False

    def __del__(self):
        pass

    def get_arch(self):
        with open(self.network, 'r') as f:
                self.params = json.load(f)
        #Mode: oneshot, construct, test
        #Alg: deterministic, random, envelopenet
        self.mode = self.params["parameters"]["mode"]
        self.algorithm = self.params["parameters"]["algorithm"]

        self.steps = self.params["parameters"]["steps"]
        self.batchsize = self.params["parameters"]["batchsize"]
        self.dataset = self.params["parameters"]["dataset"]
        self.evalintval = self.params["parameters"]["evalintval"];
        self.image_size = self.params["parameters"]["image_size"]

        self.archname=self.params["parameters"]["archname"];
        self.narchname=self.params["parameters"]["archname"];

        self.iterations = self.params["parameters"]["iterations"];
        self.initcell = self.params["initcell"]
        self.classificationcell = self.params["classificationcell"]
        if self.algorithm == "deterministic":
            self.params[self.algorithm] = {"arch": self.params["arch"]}
        if self.mode == "oneshot":
            self.iterations = 1;
        if "data_dir" in self.params["parameters"].keys():
            self.datadir = self.params["parameters"]["data_dir"]
        self.params[self.algorithm]['initcell'] = self.initcell
        self.alg = nac_alg.NACAlg(self.algorithm, self.params[self.algorithm]);
        self.arch = self.alg.generate()
        self.gpus = self.params["parameters"]["gpus"]
        print (self.arch)
        self.init_dirs(self.archname)

    def init_dirs(self, archname):
        self.traindir=self.basedir+"/train/"+archname
        self.evaldir=self.basedir+"/results/"+archname
        self.results = self.evaldir+"/res.log"
        self.results2 = self.evaldir+"/res.eval.log"
        self.network = self.evaldir+"/network.json"

    def run(self):
        i=0
        if self.count_params:
            self.train(5, redirect='>')
            return
        while (True):
            if i > int(self.iterations):
                break;
            for steps in range(int(self.evalintval), int(self.steps), int(self.evalintval)):
                if steps == int(self.evalintval):
                    self.train(steps, redirect='>')
                    self.test(redirect='>')
                else:
                    self.train(steps, redirect='>>')
                    self.test(redirect='>>')
            self.log()
            if self.mode == "construct":
                self.construct(i)
            elif self.mode == "oneshot":
                break
            i+=1

    def log(self):
       #Copy results/network files
       cmd = "cp res.log "+self.results
       os.system(cmd)
       cmd = "cp res.eval.log "+self.results2
       os.system(cmd)

    def construct(self, iteration):
        #Copy results/network files
        archstring = json.dumps(self.arch, indent=4, sort_keys=True)
        if not os.path.exists(os.path.dirname(self.network)):
            try:
                os.makedirs(os.path.dirname(self.network))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                   raise

        with open(self.network, 'w') as f:
            f.write(archstring)
        f.close()
        #Run NAC alg
        with open(self.network, 'r') as f:
            archstring = f.read()
        f.close()
        self.arch = json.loads(archstring)
        self.narch = self.alg.construct(self.arch, self.results)
        self.arch = self.narch
        if self.algorithm != "random":
             #Do not overwrite arch for random, all random nets are based on the base
             #nets, while other algs used the generated net as the next base
             self.narchname = self.archname + "-"+str(iteration+1)
        else:
             self.narchname = self.archname + "-random-"+str(iteration+1)
        print ("Constructing");
        print(self.arch)
        #archname stays the same, narchname contains name of current arch
        self.init_dirs(self.narchname)


    def train(self, steps, redirect='>>'):
        print(self.arch)
        print(self.initcell)
        print(self.classificationcell)
        cmd = "python "+self.codedir+"/nac_train.py  --train_dir="+self.traindir+" clone_on_cpu=True --dataset="+self.dataset+" --dataset_split_name=train --data_dir="+self.datadir+" --max_steps="+str(steps)+" --model_name=nacnet --arch='"+json.dumps(self.arch)+ "' --archname="+ self.narchname + " --initcell='"+json.dumps(self.initcell)+ "' --classificationcell='"+json.dumps(self.classificationcell)+"' --batch_size="+str(self.batchsize) + " --gpus=" + json.dumps(self.gpus) + " --image_size="+str(self.image_size)+" --mode="+self.mode+ " --count_params=" + str(n.count_params)+" 2"+redirect+"./res.log"
        print(cmd)
        os.system(cmd)

    def test(self, redirect='>>'):
        cmd = "python "+self.codedir+"/nac_eval.py  --eval_dir="+self.evaldir+" --checkpoint_dir="+self.traindir+" --clone_on_cpu=True  --dataset="+self.dataset+" --dataset_split_name=test --data_dir="+self.datadir+" --model_name=nacnet --arch='"+json.dumps(self.arch)+"'" + " --archname="+self.narchname + " --initcell='"+json.dumps(self.initcell)+ "' --classificationcell='"+json.dumps(self.classificationcell)+ "' --run_once=True --batch_size="+str(self.batchsize)+" --image_size="+str(self.image_size)+" --mode="+self.mode+" 2" +redirect+"./res.eval.log"
        print(cmd)
        os.system(cmd)



if __name__ == "__main__":
    n  = NAC()
    if len(sys.argv) <= 1:
        print ("Usage: python nac_gen.py <path to config file>")
        exit(-1)
    if len(sys.argv) == 3 and sys.argv[2] == '--count_params':
        n.count_params = True
    n.network = sys.argv[1]
    n.get_arch()
    n.run()
