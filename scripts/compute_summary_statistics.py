import os
import argparse
import numpy as np
import h5py as h5
from tqdm import trange
from karmma import KarmmaConfig
from karmma.utils import get_corrfunc, get_field_bins, get_1ptfunc, setup_pseudo_cls, get_pseudo_cls

def overwrite_dataset(group, key, data):
    if key in group:
        del group[key]
    group[key] = data

# Run as follows:
# config=/path/to/config.yaml
# nice -n 10 python compute_summary_statistics.py $config --recompute corr pseudo_cl 1pt

parser = argparse.ArgumentParser()
parser.add_argument('configfile')
parser.add_argument('--recompute', nargs='*', choices=['corr', 'pseudo_cl', '1pt'],
                    default=['corr', 'pseudo_cl', '1pt'])
args = parser.parse_args()

config   = KarmmaConfig(args.configfile)
analysis = config.analysis
io       = config.io
mcmc     = config.mcmc

recompute_corr      = 'corr'      in args.recompute
recompute_pseudo_cl = 'pseudo_cl' in args.recompute
recompute_1pt       = '1pt'       in args.recompute

nside     = analysis.nside
nbins     = analysis.nbins
mask      = io.mask
io_dir    = io.io_dir
n_samples = mcmc.n_samples

summary_file = os.path.join(io_dir, 'summary_statistics.h5')

print('Loading true dm maps.')
with h5.File(io.datafile, 'r') as f:
    dm_true = f['dm'][:]

print('Computing bins.')
linear_bins, log_bins = get_field_bins(dm_true, mask)

print('Setting up pseudo-Cl workspace.')
workspace, nmt_ell_bins, eff_ell, ell_edges = setup_pseudo_cls(mask)

print('Computing summary statistics for true dm.')
if recompute_corr:
    corr_true, corr_errors_true, corr_bin_centres, corr_bin_edges = get_corrfunc(dm_true, mask)
if recompute_pseudo_cl:
    pseudo_cl_true = get_pseudo_cls(dm_true, mask, nmt_ell_bins, workspace)
if recompute_1pt:
    pdf_linear_true, pdf_log_true = get_1ptfunc(dm_true, mask, linear_bins, log_bins)

if recompute_corr:
    corr_samples        = np.zeros((n_samples, nbins, nbins, len(corr_bin_centres)))
    corr_errors_samples = np.zeros_like(corr_samples)
if recompute_pseudo_cl:
    pseudo_cl_samples = np.zeros((n_samples, nbins, nbins, len(eff_ell)))
if recompute_1pt:
    pdf_linear_samples = np.zeros((n_samples, nbins, linear_bins.shape[1] - 1))
    pdf_log_samples    = np.zeros((n_samples, nbins, log_bins.shape[1] - 1))

print('Computing summary statistics for samples.')
with h5.File(os.path.join(io_dir, 'samples.h5'), 'r') as f:
    for i in trange(n_samples):
        dm_i = f['dm'][i]

        if recompute_corr:
            corr_samples[i], corr_errors_samples[i], _, _ = get_corrfunc(dm_i, mask)
        if recompute_pseudo_cl:
            pseudo_cl_samples[i] = get_pseudo_cls(dm_i, mask, nmt_ell_bins, workspace)
        if recompute_1pt:
            pdf_linear_samples[i], pdf_log_samples[i] = get_1ptfunc(dm_i, mask, linear_bins, log_bins)

print('Saving summary statistics.')
with h5.File(summary_file, 'a') as f:
    bins    = f.require_group('bins')
    truth   = f.require_group('truth')
    samples = f.require_group('samples')

    if recompute_corr:
        overwrite_dataset(bins,    'corr_bin_centres', corr_bin_centres)
        overwrite_dataset(bins,    'corr_bin_edges',   corr_bin_edges)
        overwrite_dataset(truth,   'corr',             corr_true)
        overwrite_dataset(truth,   'corr_errors',      corr_errors_true)
        overwrite_dataset(samples, 'corr',             corr_samples)
        overwrite_dataset(samples, 'corr_errors',      corr_errors_samples)

    if recompute_pseudo_cl:
        overwrite_dataset(bins,    'eff_ell',      eff_ell)
        overwrite_dataset(bins,    'ell_edges',    ell_edges)
        overwrite_dataset(truth,   'pseudo_cl',    pseudo_cl_true)
        overwrite_dataset(samples, 'pseudo_cl',    pseudo_cl_samples)

    if recompute_1pt:
        overwrite_dataset(bins,    'linear_bins',  linear_bins)
        overwrite_dataset(bins,    'log_bins',     log_bins)
        overwrite_dataset(truth,   'pdf_linear',   pdf_linear_true)
        overwrite_dataset(truth,   'pdf_log',      pdf_log_true)
        overwrite_dataset(samples, 'pdf_linear',   pdf_linear_samples)
        overwrite_dataset(samples, 'pdf_log',      pdf_log_samples)

print('Done.')