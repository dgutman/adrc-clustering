ADRC Clustering Project
==========================

Clustering dementia patients using their various tests such as clock, trail, digit span and verbal fluency tests.

Data
-------------------------
Data is located in 

`/home/mkhali8/dev/adrc-clustering/data`

The filename is

`WordFluencyMultiTest.csv` 


How to run it?
-------------------------
Clone this repo:

`git clone git@github.com:scimk/adrc-clustering.git`

cd into the root of adrc-clustering directory:

`cd adrc-clustering`

Activate the conda environment

`source /home/mkhali8/anaconda2/envs/adrc-clustering/bin/activate /home/mkhali8/anaconda2/envs/adrc-clustering/` 

Running main.py:

USAGE: ./main.py [OPTIONS]
 
The OPTIONS are:

`-k` number of clusters, default is 4

`-sse` compute sum of squared error

`-di` compute dunn index 
