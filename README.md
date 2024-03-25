# Cluster-based Personalized Federated Learning (CPFL) with CharDiv
This repo supports the paper "A Cluster-based Personalized Federated Learning Strategy for End-to-End ASR of Dementia Patients."
![CPFL_with_CharDiv_framework.png](https://github.com/user7569D/CharDiv-clustered-CPFL/blob/main/framework.png)
The cluster-based personalized federated learning (**CPFL**) strategy groups samples with similar character diversity (**CharDiv**) into cluster using K-means model $KM$, and assigns clients to train these samples federally, creating a cluster-specific model for decoding others within the same cluster.

## Environment
Use `pip -r requirements.txt` to install the same libraries

## Data preparation and preprocessing
We use [ADReSS challenge dataset](https://dementia.talkbank.org/ADReSS-2020/) as the training and testing sets. You have to join as a DementiaBank member to gain access to this dataset. Our input for ASR will be in utterance, segmented from the given session file. The information of each sample will be record in `train.csv` and `test.csv` with the following structure:
<pre><code> path, sentence
</code></pre>
where
* `path`: name of the file for the sample that ends with ".wav" and contains information for ID and the position (PAR for participant or INV for investigator) of the speaker
* `sentence`: ground truth transcription
  
A dictionary mapping  speaker ID to dementia label is also needed for analysis on separate groups of people. Generate the dictionary and assign the path to `path2_ADReSS_dict` [here](https://github.com/user7569D/CharDiv-clustered-CPFL/blob/main/src/utils.py#L81).

## Training by `bash run.sh`
The codes include training of **Fine-tuned ASR** $W_0^G$, **K-means model** $KM$, and $K$ cluster-specific models using CPFL.
1. Train Fine-tuned ASR $W_0^G$, used for extracting clustering metric
   * Important arguments
      - `FL_STAGE`: set to 1
      - `global_ep`: number of epoch for training Fine-tuned ASR $W_0^G$
      - `training_type`: only 1 (supervised) supported

2. Perform K-means Clustering, resulting in K-means model ($KM$)
   * Important arguments
      - `FL_STAGE`: set to 3
      - `training_type`: only 1 (supervised) supported
      - check the clustering metric in sections [here](https://github.com/user7569D/CharDiv-clustered-CPFL/blob/main/src/federated_main.py#L157) and [here](https://github.com/user7569D/CharDiv-clustered-CPFL/blob/main/src/federated_main.py#L219)

3. Perform CPFL, resulting in $K$ cluster-specific models
   * important arguments</summary>
      - `FL_STAGE`: set to 4
      - `training_type`: only 1 (supervised) supported
      - `N_Kmeans_update`: set using the same number as that of `epochs` to avoid re-clustering
      - `eval_mode`: set to 2 for 80% client training data and 20% client testing data, or set to 3 for 70% client training data and 10% client validation data
