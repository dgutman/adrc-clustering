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
Install Kmedoids clustering algorithm:

git clone https://github.com/scikit-learn/scikit-learn.git -b kmedoids

cd scikit-learn

python setup.py install --user

Clone this adrc clustering:

`git clone git@github.com:scimk/adrc-clustering.git`

cd into the root of adrc-clustering directory:

`cd adrc-clustering`

Rename the `config.py.example` to `config.py`

Change the `OUTPUT_DIR` to your local adrc-clustering path

Activate the conda environment

`source /home/mkhali8/anaconda2/envs/adrc-clustering/bin/activate /home/mkhali8/anaconda2/envs/adrc-clustering/` 

Enter python interactive mode:

 `python`

Install NLTK files:

`import nltk`

`nltk.download()`

Follow the instuctions for download. When given the list of packages to install. Simple install `all`

usage: ./main.py [-f FEATURES [FEATURES ...]] [-k N_CLUSTERS] [-i INPUT] [-o OUTPUT] [-sse] [-di]
 
The OPTIONS are:

`-a` clustering algorithm to use (KMeans or KMedoids)

`-d` distance metric to use with KMedoids (see [sklearn.metrics.pairwise](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html) for list of distance metrics)

`-f` list of features, space separated

`-k` number of clusters, default is 4

`-i` input file, default is specified in config.py

`-o` output directory, default is specified in config.py

`-sse` compute sum of squared error

`-di` compute dunn index 

Example usage:

`./main -k 4` generate 4 clusters

`./main.py -k 6 -sse` generate 6 clusters and sum of squared error graph

List of features
--------------------------

node_count

edge_count

cycle_count

error_count

diameter 

longest_cycle

ds_rd_t1_2_resp

ds_rd_t1_5_scr

ds_rd_t2_5_scr

np_clock_center_q13

ds_fd_t1_3_resp

ds_fd_t2_3_resp

trails_b_errors

trails_tester

avf_anm1_c

ds_fd_t2_1_resp

ds_fd_t1_5_resp

np_clock_handprop_q10

avf_anm3_c

ds_fd_t1_2_scr

np_clock_na

ds_fd_na

ds_fd_t1_6_resp

ds_fd_t1_1_resp

ds_rd_t2_4_resp

avf_anm_caferrpersev

ds_fd_t2_6_resp

np_clock_incom

ds_fd_t1_1_scr

ds_fd_t1_4_scr

np_clock_joinhand_q12

np_clock_arabnum_q2

avf_anm_ncomp

ds_rd_t1_4_resp

ds_fd_t2_1_scr

ds_fd_t2_4_resp

ds_rd_t1_1_scr

avf_veg2

avf_veg3

avf_veg1

avf_veg4

clock_file_upload

ds_rd_t1_3_resp

ds_tester

ds_fd_t2_2_scr

ds_fd_t2_6_scr

trails_b_ss

trails_sstype

ds_fd_t2_2_resp

np_clock_tester

np_clock_noextra_q11

np_clock_rpt_time

ds_rd_t2_6_scr

trails_files_upload

ds_incomp

ds_rd_span

trails_a_incomp

ds_rd_t1_5_resp

ds_fd_t2_3_scr

ds_fd_t1_6_scr

ds_fd_t2_5_resp

ds_fd_span

ds_fd_t1_3_scr

np_clock_numcirc_q6

ds_totalboth

ds_fd_t2_5_scr

np_clock_numpos_q5

ds_rd_t1_6_scr

np_clock_twohands_q7

ds_rd_na

np_clock_ordnum_q3

ds_rd_t2_6_resp

ds_rd_t1_1_resp

np_clock_norot_q4

ds_rd_t1_3_scr

trails_a_time

ds_rd_t2_1_scr

ds_fd_t1_2_resp

ds_rd_t2_2_resp

trails_a_ss

ds_rd_t1_4_scr

ds_rd_t1_6_resp

avf_onr_patient_visit

ds_rd_t2_2_scr

trails_b_time

ds_rd_t2_1_resp

np_clock_hourtarg_q8

np_clock_minute_q9

avf_anm_inc

np_clock_only12numbers_q1

ds_rd_t2_4_scr

trails_b_incomp

ds_rd_t2_3_resp

ds_fd_t2_4_scr

ds_rd_t2_5_resp

avf_anm_tester

avf_anm2_c

ds_fd_t1_4_resp

trails_b_na

trails_a_na

legacy_clock_complete

avf_anm4_c

ds_rd_t2_3_scr

trails_a_errors

ds_rd_t1_2_scr

ds_fd_t1_5_scr