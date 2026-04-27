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

"""Functions for processing confidence metrics."""

import warnings
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import scipy.special


def compute_plddt(logits: np.ndarray) -> np.ndarray:
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = scipy.special.softmax(logits, axis=-1)
  predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
  return predicted_lddt_ca * 100


def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = (breaks[1] - breaks[0])
  bin_centers = breaks + step / 2
  bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]], axis=0)
  return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: np.ndarray,
    aligned_distance_error_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Calculates expected aligned distance errors for every pair of residues.

  Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.

  Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
  return (np.sum(aligned_distance_error_probs * bin_centers, axis=-1),
          np.asarray(bin_centers[-1]))


def compute_predicted_aligned_error(
    logits: np.ndarray,
    breaks: np.ndarray) -> Dict[str, np.ndarray]:
  """Computes aligned confidence metrics from logits.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  aligned_confidence_probs = scipy.special.softmax(
      logits,
      axis=-1)
  predicted_aligned_error, max_predicted_aligned_error = (
      _calculate_expected_aligned_error(
          alignment_confidence_breaks=breaks,
          aligned_distance_error_probs=aligned_confidence_probs))
  return {
      'aligned_confidence_probs': aligned_confidence_probs,
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
  }


def _predicted_tm_term_from_logits(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: np.ndarray,
    use_full_length_for_d0: bool = False,
) -> np.ndarray:
  bin_centers = _calculate_bin_centers(breaks)
  if use_full_length_for_d0:
    num_res = logits.shape[0]
  else:
    num_res = int(np.sum(residue_weights))
  clipped_num_res = max(num_res, 19)
  d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
  probs = scipy.special.softmax(logits, axis=-1)
  tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
  return np.sum(probs * tm_per_bin, axis=-1)


def _ptm_score_from_tm_term(
    predicted_tm_term: np.ndarray,
    residue_weights: np.ndarray,
    pair_weights: np.ndarray,
) -> float:
  normed = pair_weights / (1e-8 + np.sum(pair_weights, axis=-1, keepdims=True))
  per_alignment = np.sum(predicted_tm_term * normed, axis=-1)
  best = int(np.argmax(per_alignment * residue_weights))
  return float(per_alignment[best])


def predicted_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False) -> np.ndarray:
  """Computes predicted TM alignment or predicted interface TM alignment score.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
      ipTM calculation, i.e. when interface=True.
    interface: If True, interface predicted TM score is computed.

  Returns:
    ptm_score: The predicted TM alignment or the predicted iTM score.
  """

  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])

  num_res = logits.shape[0]
  predicted_tm_term = _predicted_tm_term_from_logits(
      logits, breaks, residue_weights)

  cross_chain = np.ones((num_res, num_res), dtype=bool)
  if interface:
    cross_chain = asym_id[:, None] != asym_id[None, :]

  predicted_tm_term = predicted_tm_term * cross_chain.astype(np.float64)
  pair_weights = cross_chain.astype(np.float64) * (
      residue_weights[None, :] * residue_weights[:, None])
  return np.asarray(_ptm_score_from_tm_term(predicted_tm_term, residue_weights, pair_weights))


def predicted_tm_score_float32(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False) -> np.ndarray:
  """Computes predicted TM alignment or predicted interface TM alignment score.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
      ipTM calculation, i.e. when interface=True.
    interface: If True, interface predicted TM score is computed.

  Returns:
    ptm_score: The predicted TM alignment or the predicted iTM score.
  """
  return predicted_tm_score(logits=logits, breaks=breaks, residue_weights=residue_weights, asym_id=asym_id, interface=interface).astype('float32')

def chain_mean_cross_chain_from_pair_matrix(chain_pair_matrix: np.ndarray) -> np.ndarray:
  """For each chain, mean of off-diagonal entries (row and col mean, then mean)."""
  num_chains = chain_pair_matrix.shape[0]
  weight = np.ones((num_chains, num_chains), dtype=float)
  weight -= np.eye(num_chains, dtype=float)
  masked = np.where(weight, chain_pair_matrix, np.nan)
  xchain_row_agg = np.nanmean(masked, axis=-2)
  xchain_col_agg = np.nanmean(masked, axis=-1)
  with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    return np.nanmean(np.stack([xchain_row_agg, xchain_col_agg], axis=0), axis=0)

def _binary_interface_seq_mask(
    contact_map: np.ndarray,
    total_length: int,
    start_i: int,
    end_i: int,
    start_j: int,
    end_j: int,
    contact_threshold: float = 0.6,
) -> np.ndarray:
  contacts = np.where(contact_map[start_i:end_i + 1, start_j:end_j + 1] >=
                      contact_threshold)
  seq_mask = np.zeros(total_length, dtype=np.float64)
  if contacts[0].size == 0:
    return seq_mask

  global_i_positions = contacts[0] + start_i
  global_j_positions = contacts[1] + start_j
  global_positions = np.unique(
      np.concatenate((global_i_positions, global_j_positions)))
  seq_mask[global_positions] = 1.0
  return seq_mask

def _colabfold_actifptm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    asym_id: np.ndarray,
    residue_weights: np.ndarray,
    pair_weights: Optional[np.ndarray] = None,
) -> float:
  """Matches ColabFold's actifpTM (https://doi.org/10.1093/bioinformatics/btaf107) scoring behavior."""
  predicted_tm_term = _predicted_tm_term_from_logits(
      logits, breaks, residue_weights, use_full_length_for_d0=True)
  pair_mask = asym_id[:, None] != asym_id[None, :]
  predicted_tm_term = predicted_tm_term * pair_mask.astype(np.float64)
  if pair_weights is None:
    pair_weights = pair_mask.astype(np.float64) * (
        residue_weights[None, :] * residue_weights[:, None])
  normed = pair_weights / (1e-8 + np.sum(pair_weights, axis=-1, keepdims=True))
  per_alignment = np.sum(predicted_tm_term * normed, axis=-1)
  best = int(np.argmax(per_alignment * residue_weights))
  return float(per_alignment[best])

def chain_pair_tm_matrices(
    logits: np.ndarray,
    breaks: np.ndarray,
    asym_id: np.ndarray,
    seq_mask: Optional[np.ndarray] = None,
    contact_map: Optional[np.ndarray] = None,
    use_probs_extra: bool = False,
    contact_threshold: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns (chain_pair_iptm, chain_pair_actifptm), ordered by chain ID.

  Diagonal for chain_pair_actifptm is NaN (no self-interface). 
  If contact_map is None, chain_pair_actifptm is all NaN.
  """
  if seq_mask is None:
    residue_weights = np.ones(logits.shape[0], dtype=np.float64)
  else:
    residue_weights = (seq_mask > 0).astype(np.float64)

  unique_chains = np.unique(asym_id)
  n = len(unique_chains)
  iptm_mat = np.full((n, n), np.nan, dtype=np.float64)
  actif_mat = np.full((n, n), np.nan, dtype=np.float64)
  chain_bounds = {
      chain: (int(np.where(asym_id == chain)[0][0]), int(np.where(asym_id == chain)[0][-1]))
      for chain in unique_chains
  }

  for i, chain_i in enumerate(unique_chains):
    for j in range(i, n):
      chain_j = unique_chains[j]
      mask_union = ((asym_id == chain_i) | (asym_id == chain_j)) & (
          residue_weights > 0)
      if not np.any(mask_union):
        continue
      (idx,) = np.where(mask_union)
      ix = np.ix_(idx, idx)
      asym_s = asym_id[mask_union]
      w = residue_weights[mask_union]

      is_interface = chain_i != chain_j
      iptm_mat[i, j] = predicted_tm_score(
          logits[ix],
          breaks,
          residue_weights=w,
          asym_id=asym_s,
          interface=is_interface)
      iptm_mat[j, i] = iptm_mat[i, j]

      if contact_map is not None:
        if is_interface:
          start_i, end_i = chain_bounds[chain_i]
          start_j, end_j = chain_bounds[chain_j]
          if use_probs_extra:
            pair_residue_weights = np.zeros_like(contact_map, dtype=np.float64)
            pair_residue_weights[
                start_i:end_i + 1, start_j:end_j + 1] = contact_map[
                    start_i:end_i + 1, start_j:end_j + 1]
            pair_residue_weights[
                start_j:end_j + 1, start_i:end_i + 1] = contact_map[
                    start_j:end_j + 1, start_i:end_i + 1]
            union_mask = np.zeros(logits.shape[0], dtype=np.float64)
            union_mask[idx] = residue_weights[idx]
            actif_mat[i, j] = _colabfold_actifptm_score(
                logits,
                breaks,
                asym_id,
                union_mask,
                pair_weights=pair_residue_weights)
          else:
            active_mask = _binary_interface_seq_mask(
                contact_map,
                logits.shape[0],
                start_i,
                end_i,
                start_j,
                end_j,
                contact_threshold=contact_threshold)
            if np.any(active_mask):
              actif_mat[i, j] = _colabfold_actifptm_score(
                  logits, breaks, asym_id, active_mask)
            else:
              actif_mat[i, j] = 0.0
          actif_mat[j, i] = actif_mat[i, j]

  return iptm_mat, actif_mat

def multimer_confidence_summary(
    prediction_result: Mapping[str, Any],
    batch: Mapping[str, Any]
) -> Dict[str, Any]:
  """Multimer summary metrics for JSON export (chain-pair matrices)."""
  pae_head = prediction_result['predicted_aligned_error']
  logits = np.asarray(pae_head['logits'])
  breaks = np.asarray(pae_head['breaks'])
  asym_id = np.asarray(pae_head['asym_id']).reshape(-1)
  use_probs_extra = bool(prediction_result.get('use_probs_extra', False))
  seq_mask = batch.get('seq_mask')
  if seq_mask is not None:
    seq_mask = np.asarray(seq_mask).reshape(-1)

  ptm = float(prediction_result['ptm'])
  iptm = float(prediction_result['iptm'])

  contact_map = None
  dist = prediction_result.get('distogram')
  if dist is not None and 'logits' in dist and 'bin_edges' in dist:

    distance_threshold = 8.0
    probs = scipy.special.softmax(np.asarray(dist["logits"]), axis=-1)
    # match ColabFold's lower-edge style contact binning for actifpTM.
    contact_bins = np.concatenate([np.asarray([0.0]), np.asarray(dist["bin_edges"])], axis=0)
    contact_mass = (contact_bins < distance_threshold).astype(np.float64)
    contact_map = np.sum(probs * contact_mass[None, None, :], axis=-1)

  chain_pair_iptm, chain_pair_actifptm = chain_pair_tm_matrices(
      logits,
      breaks,
      asym_id,
      seq_mask=seq_mask,
      contact_map=contact_map,
      use_probs_extra=use_probs_extra)

  chain_ptm = np.diag(chain_pair_iptm).astype(np.float64)
  chain_iptm = chain_mean_cross_chain_from_pair_matrix(chain_pair_iptm)
  chain_actifptm = chain_mean_cross_chain_from_pair_matrix(chain_pair_actifptm)

  actifptm = np.nan
  if contact_map is not None:
    if seq_mask is None:
      w_act = np.ones(logits.shape[0], dtype=np.float64)
    else:
      w_act = (seq_mask > 0).astype(np.float64)
    if use_probs_extra:
      pw = w_act[:, None] * w_act[None, :] * (
          asym_id[:, None] != asym_id[None, :]).astype(np.float64) * contact_map
      actifptm = _colabfold_actifptm_score(
          logits, breaks, asym_id, w_act, pair_weights=pw)
    else:
      pw = np.zeros((len(asym_id), len(asym_id)), dtype=np.float64)
      unique_chains = np.unique(asym_id)
      chain_bounds = {
          chain: (int(np.where(asym_id == chain)[0][0]), int(np.where(asym_id == chain)[0][-1]))
          for chain in unique_chains
      }
      for i, chain_i in enumerate(unique_chains):
        start_i, end_i = chain_bounds[chain_i]
        for chain_j in unique_chains[i + 1:]:
          start_j, end_j = chain_bounds[chain_j]
          active_mask = _binary_interface_seq_mask(
              contact_map,
              len(asym_id),
              start_i,
              end_i,
              start_j,
              end_j)
          if np.any(active_mask):
            pw += active_mask[None, :] * active_mask[:, None]
      pw *= w_act[:, None] * w_act[None, :]
      pw *= (asym_id[:, None] != asym_id[None, :]).astype(np.float64)
      if pw.sum() >= 1e-12:
        actifptm = _colabfold_actifptm_score(
            logits, breaks, asym_id, w_act, pair_weights=pw)
      else:
        actifptm = 0.0
    if use_probs_extra and pw.sum() < 1e-12:
      actifptm = 0.0

  out = {
    'chain_ptm': chain_ptm.tolist(),
    'chain_iptm': chain_iptm.tolist(),
    'chain_pair_iptm': chain_pair_iptm.tolist(),
    'chain_actifptm': chain_actifptm.tolist(),
    'chain_pair_actifptm': chain_pair_actifptm.tolist(),
  }

  if contact_map is not None:
    out['actifptm'] = float(actifptm)
  else:
    out['actifptm'] = np.nan
  out.update({
    'iptm+ptm': float(0.8*iptm+0.2*ptm),
    'ptm': float(ptm),
    'iptm': float(iptm)
  })

  return out
