#!/usr/bin/env python

# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Union

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np
import ml_collections

# Internal import (7716).

logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2
  FIVE = 3
  TEN = 4

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - monomer model, '
                  'monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model; "multimer" computes the 3 versions of multimer models by default '
                  'if models are not specified in the --models_to_use parameter')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('end_prediction', 5, 'prediction to end with, can be used to parallelize jobs, '
                     'is combined with --start_prediction,'
                     'e.g. --start_prediction 20 --end_prediction 20 will only make prediction _20'
                     'e.g. --start_prediction 20 --end_prediction 21 will make prediction _20 and _21 etc.')
flags.DEFINE_integer('start_prediction', 1, 'prediction to start with, can be used to parallelize jobs, '
                     'is combined with --end_prediction,'
                     'e.g. --start_prediction 20 --end_prediction 20 will only make prediction _20'
                     'e.g. --start_prediction 20 --end_prediction 21 will make prediction _20 and _21 etc.')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_integer('max_recycles', 20,'Maximum number of recycles to run')
flags.DEFINE_integer('uniprot_max_hits', 50000, 'Max hits in uniprot MSA')
flags.DEFINE_integer('mgnify_max_hits', 501, 'Max hits in mgnify MSA')
flags.DEFINE_integer('uniref_max_hits', 10000, 'Max hits in uniref MSA')
flags.DEFINE_integer('bfd_max_hits', 100000, 'Max hits in BFD/uniref MSA')
flags.DEFINE_float('early_stop_tolerance', 0.5,'Early stop threshold for recycling')
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.BEST, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_list('models_to_use',None, 'Specify which neural network models in --model_preset that should be run, '
                  'each model should be formated, '
                  'for monomer and monomer_casp14 as model_X, with X the number of the model, '
                  'for monomer_ptm as model_X_ptm, with X the number of the model, '
                  'for multimer as model_X_multimer_vY with X the number of the model and Y '
                  ' the version of the model.')
flags.DEFINE_boolean('alignments_only', False, 'Whether to generate only alignments. '
                     'Only alignments will be generated by the data pipeline, '
                     'the structure inference will not be performed')
flags.DEFINE_boolean('templates', True, 'Whether to use templates or not for inference, setting it to false is faster '
                     'than filtering by date')
flags.DEFINE_boolean('dropout', False, 'Turn on dropout during inference to get more diversity')
flags.DEFINE_boolean('dropout_structure_module',False, 'Activates dropout or not during inference '
                     'in the structure module')
flags.DEFINE_string('dropout_rates_filename', None, 'Provides dropout rates for inference from a JSON file. '
                     'If None, default rates are used, if "dropout" is True.')
flags.DEFINE_float('min_score', 0, 'Predictions with a score below this threshold will be excluded from the output')
flags.DEFINE_float('stop_recycling_below', 0,
                    'After the first recycle step, only predictions with ranking confidence above this score '
                    'will continue recycling; predictions below this threshold will still be present in '
                    'ranking_debug.json and produce output.')
flags.DEFINE_float('max_score', 1,
                    'Terminates the computing process when a suitable '
                    'prediction with a ranking confidence > max_score has been obtained')
flags.DEFINE_boolean('keep_pkl', True, 'Whether to output pkl files or not.')
flags.DEFINE_boolean('reassign_chain', True, 'By default, chains IDs start from B, '
                      'activate this parameter to reassign the chains IDs from A to chain n.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax,
    alignments_only: bool):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  feature_dict = data_pipeline.process(
      input_fasta_path=fasta_path,
      msa_output_dir=msa_output_dir)
  timings['features'] = time.time() - t_0

  if alignments_only:
    logging.info('Alignments have been generated, stopping now.')
    return

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  relaxed_pdbs = {}
  relax_metrics = {}

  ranking_confidences = {}
  
  if FLAGS.model_preset == "multimer":
    iptms = {}
  ptms = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed,
                                             prediction_name=model_name)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)

    plddt = prediction_result['plddt']
    confidence = prediction_result['ranking_confidence']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']
    if FLAGS.model_preset == "multimer":
      iptms[model_name] = prediction_result['iptm'] * 1
    ptms[model_name] = prediction_result['ptm'] * 1


    if confidence >= FLAGS.min_score:

      # Remove jax dependency from results.
      np_prediction_result = _jnp_to_np(dict(prediction_result))
      if "num_recycles" in np_prediction_result:
        logging.info(f"Number of recycles for this model: {np_prediction_result['num_recycles']}")

      # Save the model outputs.
      if FLAGS.keep_pkl:
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
          pickle.dump(np_prediction_result, f, protocol=4)

      # Add the predicted LDDT in the b-factor column.
      # Note that higher predicted LDDT value means higher model confidence.
      plddt_b_factors = np.repeat(
          plddt[:, None], residue_constants.atom_type_num, axis=-1)
      unrelaxed_protein = protein.from_prediction(
          features=processed_feature_dict,
          result=prediction_result,
          b_factors=plddt_b_factors,
          remove_leading_feature_dimension=not model_runner.multimer_mode)

      unrelaxed_proteins[model_name] = unrelaxed_protein
      unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein, reassign_chain_id=FLAGS.reassign_chain)
      unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')

      with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdbs[model_name])
    else:
      print(f"Prediction {model_name} not saved, ranking confidence {confidence} \
under threshold {FLAGS.min_score}")

    if FLAGS.model_preset == "multimer":
      if prediction_result['ranking_confidence'] > FLAGS.max_score:
        print(f"\nThe max score {FLAGS.max_score} has been reached with \
  prediction {model_name}: {prediction_result['ranking_confidence']}\n")
        break

  # Rank by model confidence.
  ranked_order = [
      model_name for model_name, confidence in
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

  # Rank predictions by iptms only for multimer and ptms both
  if FLAGS.model_preset == "multimer":
    order_by_iptm = [
      model_name for model_name, iptm in
      sorted(iptms.items(), key=lambda x: x[1], reverse=True)]
  order_by_ptm = [
    model_name for model_name, ptm in
    sorted(ptms.items(), key=lambda x: x[1], reverse=True)]

  # Relax predictions.
  if models_to_relax == ModelsToRelax.BEST:
    to_relax = [ranked_order[0]]
  elif models_to_relax == ModelsToRelax.ALL:
    to_relax = ranked_order
  elif models_to_relax == ModelsToRelax.FIVE:
    to_relax = ranked_order[:5]
  elif models_to_relax == ModelsToRelax.TEN:
    to_relax = ranked_order[:10]
  elif models_to_relax == ModelsToRelax.NONE:
    to_relax = []

  for model_name in to_relax:

    if FLAGS.model_preset == "multimer":
      if model_name not in unrelaxed_proteins:
        print(f"Relax target {model_name}'s score < {FLAGS.min_score}, no output to relax.")
        break

    t_0 = time.time()
    relaxed_pdb_str, _, violations = amber_relaxer.process(
        prot=unrelaxed_proteins[model_name])
    relax_metrics[model_name] = {
        'remaining_violations': violations,
        'remaining_violations_count': sum(violations)
    }
    timings[f'relax_{model_name}'] = time.time() - t_0

    relaxed_pdbs[model_name] = relaxed_pdb_str

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(
        output_dir, f'relaxed_{model_name}.pdb')
    with open(relaxed_output_path, 'w') as f:
      f.write(relaxed_pdb_str)

  # Write out relaxed PDBs in rank order.
  for idx, model_name in enumerate(ranked_order):
    if model_name not in unrelaxed_proteins:
      continue
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if model_name in relaxed_pdbs:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  if FLAGS.model_preset == "multimer":
    with open(os.path.join(output_dir, 'ranking_iptm.json'), 'w') as f:
      f.write(json.dumps(
          {
          'iptm': iptms,
          'order': order_by_iptm,
          }, indent=4))
  with open(os.path.join(output_dir, 'ranking_ptm.json'), 'w') as f:
    f.write(json.dumps(
        {
        'ptm': ptms,
        'order': order_by_ptm
        }, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))
  if models_to_relax != ModelsToRelax.NONE:
    relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
    with open(relax_metrics_path, 'w') as f:
      f.write(json.dumps(relax_metrics, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
  _check_flag('uniref30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  _check_flag('pdb70_database_path', 'model_preset',
              should_be_set=not run_multimer_system)
  _check_flag('pdb_seqres_database_path', 'model_preset',
              should_be_set=run_multimer_system)
  _check_flag('uniprot_database_path', 'model_preset',
              should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniref30_database_path=FLAGS.uniref30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      templates=FLAGS.templates,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      mgnify_max_hits=FLAGS.mgnify_max_hits,
      uniref_max_hits=FLAGS.uniref_max_hits,
      bfd_max_hits=FLAGS.bfd_max_hits)

  end_prediction = FLAGS.end_prediction

  if run_multimer_system:
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        max_uniprot_hits=FLAGS.uniprot_max_hits)
  else:
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  if FLAGS.models_to_use:
    model_names =[m for m in model_names if m in FLAGS.models_to_use]
  if len(model_names)==0:
    raise ValueError(f'No models to run: {FLAGS.models_to_use} is not in {config.MODEL_PRESETS[FLAGS.model_preset]}')
  for model_name in model_names:
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_config.model.num_recycle = FLAGS.max_recycles
    if FLAGS.model_preset != 'multimer':
        model_config.data.common.num_recycle = FLAGS.max_recycles

    model_config.model.recycle_early_stop_tolerance = FLAGS.early_stop_tolerance
    model_config.model.stop_recycling_below = FLAGS.stop_recycling_below
    model_config.model.global_config.eval_dropout = FLAGS.dropout
    logging.info(f'Setting max_recycles to {model_config.model.num_recycle}')

    logging.info(f'Setting early stop tolerance to {model_config.model.recycle_early_stop_tolerance}')
    logging.info(f'Setting score threshold to stop recycling to {model_config.model.stop_recycling_below}')
    logging.info(f'Setting dropout to {model_config.model.global_config.eval_dropout}')

    # disabling dropout at structure module
    if not FLAGS.dropout_structure_module and FLAGS.dropout:
      model_config.model.heads.structure_module.dropout=0.0
      logging.info(f"Dropout activated but disabled for structure module.")
    # reads the dropout rates from the json file
    if FLAGS.dropout_rates_filename and FLAGS.dropout:
        with open(FLAGS.dropout_rates_filename, 'r') as f:
            dropout_dict = json.load(f)
        logging.info(f'DROPOUT rates loaded: {dropout_dict}')
        config.set_dropout_rates(model_config, dropout_dict)
        config.read_dropout_rates(model_config)


    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    for i in range(FLAGS.start_prediction, end_prediction+1):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d predictions: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=FLAGS.use_gpu_relax)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        models_to_relax=FLAGS.models_to_relax,
        alignments_only=FLAGS.alignments_only)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])

  app.run(main)
