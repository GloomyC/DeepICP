python 3.6
tensorflow >= 2.2
CUDA 10.1

--------------

DO NOT USE CONDA as linking tensorflow for compiling PointNet++ doesn't work out of box

--------------

create 
./configs/project_config.json
using
./configs/project_config_template.json
and set up your paths config file

-------------

DATASET
https://apollo.auto/southbay.html?fbclid=IwAR3qENSb6C3dDpjjeVM-Fn9kM_m7vaidhIFGzj7tzqIsQp3aOpKjQaRRfoA
download at the bottom of page, place pcds and poses in folder

--------------

depending on the way you have your cuda setup you might need to run this in terminal

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
https://stackoverflow.com/a/64472380 for more troubleshooting

--------------

to compile PointNet++ ops:
sh tf_ops/compile_ops.sh

--------------

python3 index_datset.py

--------------

to create new default model

python3 train.py -mn <model_name> -new

--------------

for for more options

python3 train.py --help

create new configs in their respecive folders to use them as arguments











