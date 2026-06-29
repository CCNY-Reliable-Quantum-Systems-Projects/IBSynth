# IBSynth
IBSynth performs multi round re-synthesis leveraging BQSkit methods.

## Installation
Clone the repository, cd into the repo directory and just run ```pip install -e .```.

There are quite a few dependencies, the major ones are:
- bqskit
- qiskit
- torch

## Execution
Use the main execution function provided to run our proposed methods found in IBSynth.interblock.exec.run_interblock(). From this method you can set things like seed and error cap and you can define your own workflow pass you want to have ran over the starting partitioned circuit and subsequent interblocks. This workflow defines the size of the parititoned blocks, the error threshold allowed by Qsearch (default: 1e-8), etc. In the pass you can also replace the ForEachBlockPass with NNForEachBlockPass which will leverage the ml classifier.

## Side-Notes
BQSkit heuristic synthesis uses your machine with the seed to implement psuedo-randomness for reproducability. In our paper we use seeds (0, 1, 2). As a result it will be hard reproducing the results found in our paper exactly. Generally you should achieve similar results, though because our method relies heavily on the structure of synthesized blocks, there are outlier runs which may perform much better or worse than the runs on average... be mindful of these cases.