![header](imgs/header.png)

# MassiveFold

Table of contents
=================

  * [Setup](#setup)
  * [Added parameters](#added-parameters)
  * [Dropout](#dropout)
  * [Usage](#usage)
    * [Examples](#example)
  * [MassiveFold in parallel](#running-massivefold-in-parallel)
      * [Setup](#setup-1)
      * [Header Building](#jobfiles-header-building)
      * [Usage](#usage-1)
      * [Inference workflow](#inference-workflow)
  * [Plots](#mf_plots-output-representation)


This AlphaFold version aims at massively expanding the sampling of structure predictions following Björn Wallner's 
AFsample version of AlphaFold (https://github.com/bjornwallner/alphafoldv2.2.0/)  and to provide some optimizations in the computing.  
In particular, it optimizes the parallellization of the computing, generating automatically batches of predictions that can be run in parallel. 
The size of these batches is automatically calculated running a first calibrating run or set manually. All the results are then gathered 
in a same folder and a final global ranking is performed on all the produced structures.
The optimizations and the parameters added to the genuine DeepMind's AlphaFold are described below.

MassiveFold is an extended version of DeepMind's AlphaFold v2.3.2: https://github.com/deepmind/alphafold

# Setup
The setup is the same as the one for AlphaFold v2.3 except that this repository has to be used instead of the DeepMind's one.  
We use an installation based on conda. You can install it following these steps https://github.com/kalininalab/alphafold_non_docker
or using the conda environment file that we provide (env.yml). In this last case, don't forget to apply the OpenMM patch and 
to add the chemical properties to the common folder.

```
conda create env -f env.yml
conda activate massivefold-1.1.0
cd ${CONDA_PREFIX}/lib/python3.8/site-packages/
wget -N https://raw.githubusercontent.com/GBLille/MassiveFold/MFv1.1.0/docker/openmm.patch
patch -p0 -N < openmm.patch
cd ${CONDA_PREFIX}/lib/python3.8/site-packages/alphafold/common/
wget -N https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

However, v1 and v2 neural network (NN) model parameters have to be present in the *param* folder and should contain the 
version number in their name.\
Therefore, the list of NN model parameters in the folder should be as follows:

params_model_1_multimer_v1.npz  
params_model_1_multimer_v2.npz  
params_model_1_multimer_v3.npz  
params_model_1.npz  
params_model_1_ptm.npz  
params_model_2_multimer_v1.npz  
params_model_2_multimer_v2.npz  
params_model_2_multimer_v3.npz  
params_model_2.npz  
params_model_2_ptm.npz  
params_model_3_multimer_v1.npz  
params_model_3_multimer_v2.npz  
params_model_3_multimer_v3.npz  
params_model_3.npz  
params_model_3_ptm.npz  
params_model_4_multimer_v1.npz  
params_model_4_multimer_v2.npz  
params_model_4_multimer_v3.npz  
params_model_4.npz  
params_model_4_ptm.npz  
params_model_5_multimer_v1.npz  
params_model_5_multimer_v2.npz  
params_model_5_multimer_v3.npz  
params_model_5.npz  
params_model_5_ptm.npz  

Parameters for monomer and multimer v3 are available here: https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar  
Parameters for monomer and multimer v2 are available here: https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar  
Parameters for monomer and multimer v1 are available here: https://storage.googleapis.com/alphafold/alphafold_params_2021-10-27.tar  

# Added parameters
This is the list of the parameters added to AlphaFold 2.3.2 and their description, also accessible through the --help option.

  **--alignments_only**: whether to generate only alignments. Only alignments will be generated by the data pipeline, the modelling will not be performed
    (default: 'false')  
  **--dropout**: turn on drop out during inference to get more diversity
    (default: 'false')  
  **--dropout_structure_module**: activates dropout or not during inference  
  &nbsp;&nbsp;&nbsp;&nbsp; in the structure module to get more diversity (default: 'false')  
  **--dropout_rates_filename**: provide dropout rates at inference from a json file.
  If None, default rates are used, if "dropout" is True.  
  **--max_recycles**: maximum number of recycles to run
    (default: '20')
    (an integer)  
  **--early_stop_tolerance**: early stopping threshold for recycling
    (default: '0.5')
    (a number)  
  **--bfd_max_hits**: max hits in BFD/uniref MSA
    (default: '100000')
    (an integer)  
  **--mgnify_max_hits**: max hits in mgnify MSA
    (default: '501')
    (an integer)  
 **--uniprot_max_hits**: max hits in uniprot MSA
    (default: '50000')
    (an integer)  
  **--uniref_max_hits**: max hits in uniref MSA
    (default: '10000')
    (an integer)  
  **--model_preset**: <monomer|monomer_casp14|monomer_ptm|multimer>:  
  &nbsp;&nbsp;&nbsp;&nbsp; choose preset model configuration - monomer model,  
  &nbsp;&nbsp;&nbsp;&nbsp; monomer model with extra ensembling, monomer model with pTM head, or  
  &nbsp;&nbsp;&nbsp;&nbsp; multimer model; "multimer" computes the 3 versions of multimer models by default  
  &nbsp;&nbsp;&nbsp;&nbsp; if models are not specified in the *--models_to_use* parameter  
  &nbsp;&nbsp;&nbsp;&nbsp; (default: 'monomer')  
  **--models_to_use**: specify which models in *--model_preset* that should be run, each model should be formated,  
  &nbsp;&nbsp;&nbsp;&nbsp; for monomer and monomer_casp14 as model_X, with X the number of the model,  
  &nbsp;&nbsp;&nbsp;&nbsp; for monomer_ptm as model_X_ptm, with X the number of the model,  
  &nbsp;&nbsp;&nbsp;&nbsp; for multimer as model_X_multimer_vY with X the number of the model and Y  
  &nbsp;&nbsp;&nbsp;&nbsp; the version of the model.')  
  &nbsp;&nbsp;&nbsp;&nbsp; (a comma separated list)  
  **--start_prediction**: model to start with, can be used to parallelize jobs,  
  &nbsp;&nbsp;&nbsp;&nbsp; *e.g.* --num_predictions_per_model 20 --start_prediction 20 will only make model _20  
  &nbsp;&nbsp;&nbsp;&nbsp; *e.g.* --num_predictions_per_model 21 --start_prediction 20 will make model _20 and _21 *etc.*  
  &nbsp;&nbsp;&nbsp;&nbsp; (default: '1')  
  **--end_prediction**: index of the last predicted structure,  
&nbsp;&nbsp;&nbsp;&nbsp; also designates how many predictions (each with a different random seed) will be  
&nbsp;&nbsp;&nbsp;&nbsp; generated per model. *e.g.* if this is 2 and there are 5 models then there will be 10 predictions per input.  
&nbsp;&nbsp;&nbsp;&nbsp; Note: this FLAG works for monomer and multimer  
&nbsp;&nbsp;&nbsp;&nbsp; (default: '5')  
  **--templates**: whether to use templates or not, setting it to false is faster than filtering by date
  (default: 'true')  
  **--score_threshold_output**: only predictions with ranking confidence above this score  
  &nbsp;&nbsp;&nbsp;&nbsp; will generate pdb and pkl files, predictions below this  
  &nbsp;&nbsp;&nbsp;&nbsp; threshold will still be present in ranking_debug.json.  
  &nbsp;&nbsp;&nbsp;&nbsp; (default: '0')  
  **--max_score_stop**: stops the computing process when a suitable  
  &nbsp;&nbsp;&nbsp;&nbsp; prediction with a ranking confidence > max_score_stop has been obtained  
  &nbsp;&nbsp;&nbsp;&nbsp; (default: '1')

# Dropout
The dropout at inference can be activated with the **--dropout** parameter set to true for activating it in the Evoformer 
module and **--dropout_structure_module** set to true for activating it in the structure module.
The same dropout rates as those used by DeepMind at training are used. Here are DeepMind's architectural details 
(Jumper J et al, Nature, 2021 - Fig 3.a), annotated by Björn Wallner for CASP15 (https://predictioncenter.org/), 
that shows the various dropout rates:  

![Dropout](imgs/dropout_arch.png)

However, the **--dropout_rates_filename** parameter allows to modify these rates, providing them in a json file. 
Here is an example of the content of such a json file:
```json
{  
    "dropout_rate_msa_row_attention_with_pair_bias": 0.15,  
    "dropout_rate_msa_column_attention": 0.0,  
    "dropout_rate_msa_transition": 0.0,  
    "dropout_rate_outer_product_mean": 0.0,  
    "dropout_rate_triangle_attention_starting_node": 0.25,  
    "dropout_rate_triangle_attention_ending_node": 0.25,  
    "dropout_rate_triangle_multiplication_outgoing": 0.25,  
    "dropout_rate_triangle_multiplication_incoming": 0.25,  
    "dropout_rate_pair_transition": 0.0,  
    "dropout_rate_structure_module": 0.1  
}  
```

# Usage
## Example
By default, MassiveFold runs with the same parameters as AlphaFold2, except it uses all the versions 
of neural network model parameters for complexes and not only the ones of the last version.  

Here is an example how to run a multimer prediction with all versions of neural network model parameters, without templates,
activating dropout at inference in both Evoformer and structure module, with 100 recycles maximum and early stop tolerance 
set to 0.1 Angströms. 

```bash
python3 ./run_alphafold.py
    --fasta_paths=seq.fasta
    --output_dir=./output
    --data_dir=*path_to_set*
    --db_preset=full_dbs
    --model_preset=multimer
    --max_template_date=2023-07-11
    --use_precomputed_msas=false
    --models_to_relax=best
    --use_gpu_relax=true
    --alignments_only=false
    --dropout=true
    --dropout_structure_module=true
    --dropout_rates_filename=
    --max_recycles=100
    --early_stop_tolerance=0.1
    --bfd_max_hits=100000
    --mgnify_max_hits=501
    --uniprot_max_hits=50000
    --uniref_max_hits=10000
    --models_to_use=
    --start_prediction=1
    --end_prediction=5
    --templates=false
    --score_threshold_output=0
    --max_score_stop=1
    --uniref90_database_path=*path_to_set*
    --mgnify_database_path=*path_to_set*
    --template_mmcif_dir=*path_to_set*
    --obsolete_pdbs_path=*path_to_set*
    --bfd_database_path=*path_to_set*
    --pdb_seqres_database_path=*path_to_set*
    --uniref30_database_path=*path_to_set*
    --uniprot_database_path=*path_to_set*
```

To only use a selection of models, separate them with a comma in the ***--models_to_use*** parameter, *e.g.*:  
--models_to_use=model_3_multimer_v1,model_3_multimer_v3  

A script is also provided to relax only one structure. The pkl file of the prediction has to be given in parameters and the 
*features.pkl* file must be present in the folder. *e.g.*:
```bash
python3 run_relax_from_results_pkl.py result_model_4_multimer_v3_pred_0.pkl
```
# Running MassiveFold in parallel
MassiveFold is designed for an optimized use on a GPU cluster because it can automatically split a prediction run in many jobs. 
This automatic splitting is also convenient for runs on a simple GPU server. All the developments were made to be used 
with a **SLURM** workload manager, but can be adapted to any other resource managing system working with job arrays, 
simply modifying the **header** files. The following diagram shows how MassiveFold runs in parallel.

![header](imgs/massivefold_diagram.svg)

A run is composed of three steps:  
1. **alignment**: on CPU, sequence alignments is the initiation step (can be skipped if alignments are already computed)

2. **structure prediction**: on GPU, structure predictions follows the massive sampling principle. The total number 
of predictions is divided into smaller batches and each of them is distributed on a single GPU. These jobs wait for the 
alignment job to be over, if the alignments are not provided by the user.

3. **post_treatment**: on CPU, it finishes the job by gathering all batches outputs and produces plots with the 
[MF_plots module](#mf_plots-output-representation) to visualize the run's performances. This jobs is executed only once 
all the structure predictions are over. 

To run MassiveFold in parallel, you can use the parallelization module in the **MF_scripts** directory.

## Setup

1. File architecture

The file tree that the paths in the following step follow looks like this:
```bash
.
├── MassiveFold
└── massivefold_runs
    ├── input
        └── test_multimer.fasta
    ├── log
    │   └── test_multimer/basic
    │       ├── jobarray_0.log
    │       ...
    │       └── post_treatment.log
    ├── output
    │   └── test_multimer
    │       ├── basic
    │       │   ├── ranked_0_unrelaxed_model_x_multimer_vi_pred_n.pdb
    │       │   ...
    │       │   └── ranking_debug.json
    │       └── msas
    └── scripts
        ├── headers/
        ├── templates/
        ├── MF_parallel.sh
        ├── batching.py
        ├── create_jobfile.py
        ├── examine_run.py
        ├── params.json
        ├── get_batch.py
        └── organize_outputs.py
```
To set up this file tree, follow these instructions:
```bash
# Chose where to install your massivefold runs directory
./install.sh <INSTALLATION_PATH> <data_dir>
```
The **data_dir** parameter should be the path used in AlphaFold2 installation where the databases are downloaded.

2. Create header files  

Refer to [Jobfile header building](#jobfiles-header-building) for this installation step.

To run MassiveFold in parallel on your cluster/server, it is **required** to build custom jobfile headers for each step.  
They have to be added in the path set in the **\<INSTALLATION_PATH>/scripts/headers/** directory that can be modified in the "jobfile_headers_dir" parameter of the params.json file.  
Headers for Jean Zay cluster are provided as examples to follow (named *example_header_\<step>_jeanzay.slurm*).  


3. Set custom parameters

Edit the json parameter located in <INSTALLATION_PATH>/scripts/params.json. In our example:

```bash
cd <INSTALLATION_PATH>/massivefold_runs/scripts/
```
And modify params.json:
```json
 "custom_params": {
    "run_massivefold": "../MassiveFold_dev/run_alphafold.py",
    "run_massivefold_plots": "../MassiveFold_dev/MF_scripts/plots/MF_plots.py",
    "jobfile_headers_dir": "./headers",
    "jobfile_templates_dir": "./templates",
    "output_dir": "../output",
    "logs_dir": "../log",
    "data_dir": "AlphaFold2 <DOWNLOAD_DIR>"
 },
...
```
The **data_dir** parameter should be the path used in AlphaFold2 installation where the databases are downloaded.

4. Launch a run

Set the parameters of your run in the **MF_run** section of the **params.json** file, then run massivefold in parallel with, for instance:
```bash
./MF_parallel -s ../input/test_multimer.fasta -r basic_run -p 67 -f params.json
```
For more help, run:
```bash
./MF_parallel -h
```

### Jobfile's header building

The jobfile templates for each step are built by combining the jobfile header that you have to create in 
**MF_scripts/parallelization/headers** with the jobfile body in **MF_scripts/parallelization/templates/**.

Only the headers have to be adapted in function of your computing infrastructure. 
Each of the three headers (alignment, jobarray and post treatment) must be located in the **headers** directory 
inside the **pipeline** directory (see [File architecture](#setup-1) section).

Their names should be identical to:
* **header_alignment.slurm**
* **header_jobarray.slurm**
* **header_post_treatment.slurm**

The templates work with the parameters provided in **run_params.json** given as a parameter to the **MF_parallel.sh** script.\
These parameters are substituted in the template job files thanks to the python library [string.Template](https://docs.python.org/3.8/library/string.html#template-strings).\
Refer to [How to add a parameter](#how-to-add-a-parameter) for parameter substitution.

- **Requirement:** In the jobarray header should be stated that it is a job array and the number of tasks in it has to 
be given.\
The task number argument is substituted with the *$substitute_batch_number* parameter.\
For slurm, the expression should be:
```bash
#SBATCH --array=0-$substitute_batch_number
```
For example, if there are 45 batches, with one batch per task of the job array, the substituted expression will be:
```bash
#SBATCH --array=0-44
```

- To store jobfile logs while following [Setup](#setup-1)'s file tree, add these lines in the headers:

In **header_alignment.slurm**:
```bash
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/alignment.log
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/alignment.log
```
In **header_jobarray.slurm**:

```bash
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log
```
In **header_post_treatment.slurm**:
```bash
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/post_treatment.log
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/post_treatment.log
```
On the git, we provide templates for the Jean Zay french CNRS national GPU cluster accessible at the 
[IDRIS](http://www.idris.fr/).

#### How to add a parameter
- Add **\$new_parameter** or **\$\{new_parameter\}** in the template where you want its value to be set and in the 
"custom_params" section of **run_params.json** where its value can be specified and changed for each run.

**Example** in the json parameters file for Jean Zay headers:
```json
...
  "custom_params": {
      "jeanzay_account": "project@v100",
      "jeanzay_gpu_with_memory": "v100-32g",
      "jeanzay_jobarray_time": "10:00:00"
  },
...
```
Where project is your 3 letter project with allocated hours on Jean Zay.

- These parameters will be substitued in the header where the parameter keys are located:

```bash
#SBATCH --account=$jeanzay_account

#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=1
#SBATCH --array=0-$substitute_batch_number
#SBATCH --time=$jeanzay_jobarray_time
##SBATCH --qos=qos_gpu-dev             # Uncomment for job requiring less than 2 hours
##SBATCH --qos=qos_gpu-t4         # Uncomment for job requiring more than 20h (max 16 GPUs)
#SBATCH -C $jeanzay_gpu_with_memory             # Use gpu
```
- Never use single \$ symbol for other uses than parameter/value substitution from the json file.\
To use $ inside the template files (bash variables or other uses), use instead $$ as an escape following 
[string.Template](https://docs.python.org/3.8/library/string.html#template-strings) documentation.

## Usage

```bash
./MF_parallel.sh -s sequence_path -r run_name -p predictions_per_model -f parameters_file 
```
Other facultative parameters can be set and can be consulted with:

```bash
./MF_parallel.sh -h
```
This is the help message associated with this command:

```txt
Usage: ./MF_parallel.sh -s str -r str -p int -f str [-m str] [-n str] [-b int | [[-C str | -c] [-w int]] ]
./MF_parallel.sh -h for more details 
  Required arguments:
    -s| --sequence: name of the sequence to infer, same as input file without '.fasta'.
    -r| --run: name chosen for the run to organize in outputs.
    -p| --predictions_per_model: number of predictions computed for each neural network model.
    -f| --parameters: json file's path containing the parameters used for this run.

  Facultative arguments:
    -b| --batch_size: number of predictions per batch, should not be higher than -p (default: 25).
    -m| --msas_precomputed: path to directory that contains computed msas.
    -n| --top_n_models: uses the 5 models with best ranking confidence from this run's path.
    -w| --wall_time: total time available for calibration computations, unit is hours (default: 20).
    -C| --calibration_from: path of a previous run to calibrate the batch size from (see --calibrate).

  Facultative options:
    -c| --calibrate: calibrate --batch_size value. Searches for this sequence previous runs and uses
        the longest prediction time found to compute the maximal number of prediction per batch.
        This maximal number depends on the total time given by --wall_time.
```
### Inference workflow

It launches MassiveFold with the same parameters introduced above but instead of running **run_alphafold.py** script a single time, 
it divides it in multiple batches.


For the following examples, we assume that **--model_preset=multimer** as it is the majority of cases to run MassiveFold in parallel.

However, **--model_preset=monomer_ptm** works too and needs to be adapted accordingly, at least the models to use.

You can decide how the run will be divided by assigning **MF_parallel.sh** parameters *e.g.*:

```bash
./MF_parallel.sh -s ..input/H1144.fasta -r 1005_preds -p 67 -b 25 -f run_params.json
```

The predictions are computed individually for each neural network model,  **-p** or **--predictions_per_model** allows to specify 
the number of predictions desired for each chosen model.\
These **--predictions_per_model** are then divided into batches with a fixed **-b** or **--batch_size** to optimize the run 
in parallel as each batch can be computed on a different GPU, if available. The last batch of the cycle is generally smaller 
than the others to match the number of predictions fixed by **--predictions_per_model**.

***N.B.***: an interest to use MF_parallel.sh on a single server with a single GPU is to be able to run massive sampling for a 
structure in low priority, allowing small jobs with higher priority to be run in between.

For example, with **-b 25** and **-p 67** the predictions are divided into the following batches, which is repeated 
for each NN model:

  1.  First batch: **--start_prediction=0** and **--num_predictions_per_model=24**
  2.  Second batch: **--start_prediction=25** and **--num_predictions_per_model=49**
  3.  Third batch: **--start_prediction=50** and **--num_predictions_per_model=67** 

By default (if **--models_to_use** is not assigned), all NN models are used: with **--model_preset=multimer**, 15 models 
in total = 5 neural network models $\times$ 3 AlphaFold2 versions.

The prediction number per model can be adjusted, here with 67 per model and 15 models, it amounts to **1005 predictions 
in total divided into 45 batches**.

### Run parameters

#### Parameters in MF_parallel.sh

In addition to the parameters displayed with -h option, the parameters file set in **-f** or **--parameters** should be 
organized as *MF_scripts/parallelization/generic_params.json*.

#### Parameters in the json file

Each section of **run_params.json** is used for a different purpose.

The section **MF_parallel** designates the parameters relative to the whole run.  

It is presented as:

```json
   "MF_parallel": {
       "run_massivefold": "<cloned_repo_path>/run_alphafold.py",
       "run_massivefold_plots": "<cloned_repo_path>/MF_scripts/plots/MF_plots.py",
       "data_dir": "<data_dir>",
       "jobfile_header_dir": "<INSTALLATION_PATH>/headers/",
       "jobfile_templates_dir": "<INSTALLATION_PATH>/templates",
       "output_dir": "../output",
       "logs_dir": "../log",
       "predictions_to_relax": "",
       "models_to_use": ""
   },
...
```
You have to fill the paths in this section. Templates are specified here to setup the run, build your owns according to the [Template building](#jobfiles-header-building) section.

- The **custom_params** section is relative to the personalized parameters that you want to add for your own cluster. 
For instance, for the Jean Zay GPU cluster:
```json
...
  "custom_params": 
    {
      "jeanzay_project": "project",
      "jeanzay_account": "project@v100",
      "jeanzay_gpu_with_memory": "v100-32g",
      "jeanzay_alignment_time": "10:00:00",
      "jeanzay_jobarray_time": "10:00:00"
    }
...
```
As explained in [How to add a parameter](#how-to-add-a-parameter), these variables are substituted by their value 
when the jobfiles are created.

- The **MF_run** section gathers all the parameters used by MassiveFold for the run. (see [Example](#example) and 
[Added parameters](#added-parameters)). All parameters  except *--models_to_relax*, *--use_precomputed_msas*, *--alignment_only*, *--start_prediction*, *--end_prediction*, *--fasta_path* and *--output_dir* are exposed in this
section.\
You can adapt the parameters values in function of your needs.\
The non exposed parameters mentioned before are set in intern by the MF_parallel.sh pipeline.

```json
... 
 "MF_run": {
    "MF_run_model_preset": "multimer",
    "MF_run_dropout": "false",
    "MF_run_dropout_structure_module": "false",
    "MF_run_dropout_rates_filename": "",
    "MF_run_templates": "true",
    "MF_run_min_score": "0",
    "MF_run_max_batch_score": "1",
    "MF_run_max_recycles": "20",
    "MF_run_db_preset": "full_dbs",
    "MF_run_use_gpu_relax": "true",
    "MF_run_models_to_relax": "none",
    "MF_run_early_stop_tolerance": "0.5",
    "MF_run_bfd_max_hits": "100000",
    "MF_run_mgnify_max_hits": "501",
    "MF_run_uniprot_max_hits": "50000",
    "MF_run_uniref_max_hits": "10000"
},
...
```
Lastly, section **MF_plots** is used for the MassiveFold plotting module.

```json
...
  "MF_plots": 
    {
      "MF_plots_top_n_predictions":"5",
      "MF_plots_chosen_plots": "coverage,DM_plddt_PAE,CF_PAEs"
    }
```
# massivefold_plots: output representation

MassiveFold plotting module can be used on a MassiveFold output to represent its results efficiency.  

Here is an example of a basic command you can run:
```bash
./massivefold_plots.py --input_path=<path_to_MF_output> --chosen_plots=DM_plddt_PAE
```
## Required arguments
- **--input_path**: it designates MassiveFold output dir and the directory to store the plots except if you want them in a separate directory, use *--output_path* for this purpose

- **--chosen_plots**: plots you want to get. You can give a list of plot names separated by a coma (e.g: *--chosen_plots=coverage,DM_plddt_PAE,CF_PAEs*). This is the list of all the plots available:
  * DM_plddt_PAE: Deepmind's plot for predicted lddt per residue and predicted aligned error matrix
  ![header](imgs/plot_illustrations/plddt_PAES.png)
  * CF_plddt:
  * CF_PAEs: Colabfold's plot for predicted aligned error of the n best predictions set with *--top_n_predictions*
  ![header](imgs/plot_illustrations/PAEs.png)
  * CF_plddts:
  ![header](imgs/plot_illustrations/plddts.png)
  * coverage: Colabfold's plot for sequence alignment coverage
  ![header](imgs/plot_illustrations/coverage.png)
  * score_distribution:
  * distribution_comparison: ranking confidence distribution comparison between different massivefold outputs, tipically useful for runs with different parameters on the same sequence.
  ![header](imgs/plot_illustrations/distribution_comparison.png)
  * recycles: ranking confidence during the recycle process
  ![header](imgs/plot_illustrations/recycles.png)

## Facultative arguments
- *--top_n_predictions*: (default 10), number of predictions taken into account for plotting, it will be the n best predictions
- *--runs_to_compare*: names of the runs you want to compare on their distribution,this argument is coupled with **--chosen_plots=distribution_comparison**

# Authors
Guillaume Brysbaert (UGSF - UMR 8576, France)  
Nessim Raouraoua (UGSF - UMR 8576, France)  
Marc F Lensink (UGSF - UMR8576, France)  
Christophe Blanchet (IFB, France)  
Claudio Mirabello (NBIS, Sweden)  
Björn Wallner (Linköping University, Sweden)  

This work was carried out as part of the Work Package 4 of the [MUDIS4LS project](https://www.france-bioinformatique.fr/actualites/mudis4ls-le-projet-despaces-numeriques-mutualises-pour-les-sciences-du-vivant/) 
lead by the French Bioinformatics Institute ([IFB](https://www.france-bioinformatique.fr/)). It was initiated at the [IDRIS Open Hackathon](http://www.idris.fr/annonces/idris-gpu-hackathon-2023.html), 
part of the Open Hackathons program. The authors would like to acknowledge OpenACC-Standard.org for their support.
