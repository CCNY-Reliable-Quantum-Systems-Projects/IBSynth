# IBSynth
IBSynth performs multi round re-synthesis leveraging BQSkit methods.

## Installation
Clone the repository, cd into the repo directory and just run ```pip install -e .```.

There are quite a few dependencies, the major ones are:
- bqskit
- qiskit
- torch

## Execution
Use the main execution functions provided to run our proposed methods found in InterBlock.exec. From these methods you can set things like seed and error cap.

## Side-Notes
BQSkit heuristic synthesis uses your machine with the seed to implement psuedo-randomness for reproducability. As a result it will be hard reproducing the results found in our paper exactly. Generally you should achieve similar results, though because our method relies heavily on the structure of synthesized blocks, there are outlier runs which may perform much better or worse than the runs on average... be mindful of these cases