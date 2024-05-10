#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import scvi
import scanpy as sc
import time

from anndata import AnnData
from scipy import linalg
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, sparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, r2_score
from sklearn.preprocessing import label_binarize
from math import sqrt
from scipy.stats import gaussian_kde

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import argparse
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import QED
from tqdm import tqdm
sys.path.append('../')
from perturbnet.perturb.util import * 
from perturbnet.perturb.cinn.module.flow import * 
from perturbnet.perturb.genotypevae.genotypeVAE import *
from perturbnet.perturb.data_vae.modules.vae import *
from perturbnet.perturb.cinn.module.flow_generate import SCVIZ_CheckNet2Net
from matplotlib.colors import ListedColormap




def Seq_to_Embed_ESM(ordered_trt, batch_size, model, alphabet, save_path = None):
    data = []
    count = 1
    batch_converter = alphabet.get_batch_converter()
    for i in ordered_trt:
        data.append((count,i))
        count += 1

    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    for j in tqdm(range(len(batches))):
        batch = batches[j]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len + 1].mean(0).numpy().reshape(1, -1))
    if save_path:
        np.save(save_path,sequence_representations)
    
    return(sequence_representations)


def create_train_test_splits_by_key(adata, train_ratio, add_key, split_key, control, random_seed=None):
    """
    Splits the observations in an AnnData object into training and testing sets based on unique values of a specified key,
    with certain control values always included in the training set.
    
    Parameters:
    adata (AnnData): The AnnData object containing the dataset.
    train_ratio (float): The proportion of unique values to include in the train split (0 < train_ratio < 1), excluding control values.
    add_key (str): The key to be added to adata.obs, where the train/test labels will be stored.
    split_key (str): The key in adata.obs used to determine the unique values for making splits.
    control (list): A list of values from the split_key that should always be included in the training set.
    random_seed (int, optional): The seed for the random number generator for reproducibility. If None, the seed is set based on the current time.

    Returns:
    None: The function adds a new column to adata.obs indicating train/test split.
    """
    if random_seed is None:
        random_seed = int(time.time())
    np.random.seed(random_seed)
    unique_values = adata.obs[split_key].unique()
    non_control_values = [value for value in unique_values if value not in control]
    np.random.shuffle(non_control_values)
    # Calculate the number of unique non-control values to include in the train set
    num_train = int(np.floor(train_ratio * len(non_control_values)))
    # Select training values from non-control values
    train_values = set(non_control_values[:num_train])
    # Combine train values with control values
    train_values.update(control)
    # Assign 'train' or 'test' to the observations based on the split of unique values
    adata.obs[add_key] = adata.obs[split_key].apply(lambda x: 'train' if x in train_values else 'test')

    
def build_cinn(adata, cell_repre_model, perturbation_key,  perturbation_type = ["chem", "genetic", "protein"], 
               trt_key = "ordered_all_trt", embed_key = "ordered_all_embedding", 
               random_seed = 42):
    if perturbation_type == "protein":
        scvi_model = cell_repre_model
        perturb_with_onehot = np.array(adata.obs[perturbation_key])
        trt_list = np.unique(perturb_with_onehot)
        embed_idx = []
        for i in range(len(trt_list)):
            trt = trt_list[i]
            idx = np.where(adata.uns[trt_key] == trt)[0][0]
            embed_idx.append(idx)
        embeddings = adata.uns[embed_key][embed_idx]
        
        perturbToEmbed = {}
        for i in range(trt_list.shape[0]):
            perturbToEmbed[trt_list[i]] = i
        torch.manual_seed(42)
        
        flow_model = ConditionalFlatCouplingFlow(conditioning_dim=1280,
                                    embedding_dim=10,
                                    conditioning_depth=2,
                                     n_flows=20,
                                     in_channels=10,
                                     hidden_dim=1024,
                                     hidden_depth=2,
                                     activation="none",
                                     conditioner_use_bn=True)
        model_c = Net2NetFlow_scVIFixFlow(configured_flow = flow_model,
                               cond_stage_data = perturb_with_onehot,
                               perturbToEmbedLib = perturbToEmbed,
                               embedData = embeddings,
                               scvi_model = cell_repre_model)
        return model_c, embeddings, perturbToEmbed
    
    
def predict_perturbation_protein(perturbnet_model, perturbation_embeddings, library_latent, n_cell = 100, random_seed = 42):
    np.random.seed(random_seed)
    Lsample_idx = np.random.choice(range(library_latent.shape[0]), n_cell, replace=True)
    onehot_indice_trt = np.tile(perturbation_embeddings, (n_cell, 1))
    trt_onehot = onehot_indice_trt + np.random.normal(scale = 0.001, size = onehot_indice_trt.shape)
    library_trt_latent = library_latent[Lsample_idx]
    fake_latent, fake_data = perturbnet_model.sample_data(trt_onehot, library_trt_latent)
    
    return fake_latent, fake_data, trt_onehot


def umapPlot_latent_check(real_latent, fake_latent, path_file_save = None):
    all_latent = np.concatenate([fake_latent, real_latent], axis = 0)
    cat_t = ["Real"] * real_latent.shape[0]
    cat_g = ["Fake"] * fake_latent.shape[0]
    cat_rf_gt = np.append(cat_g, cat_t)
    trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(all_latent)
    X_embedded_pr = trans.transform(all_latent)
    df = X_embedded_pr.copy()
    df = pd.DataFrame(df)
    df['x-umap'] = X_embedded_pr[:,0]
    df['y-umap'] = X_embedded_pr[:,1]
    df['category'] = cat_rf_gt
    
    chart_pr = ggplot(df, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
    + geom_point(size=0.5, alpha = 0.5) \
    + ggtitle("UMAP dimensions")

    if path_file_save is not None:
        chart_pr.save(path_file_save, width=12, height=8, dpi=144)
    return chart_pr

def contourplot_space_mapping_pca(embeddings_cell, embeddings_pert, background_pert, background_cell, highlight_label,
                                  random_state=42, n_pcs=50, bandwidth=0.2):
    # Apply PCA to reduce the dimensions of perturbations to 50 principal components
    pca = PCA(n_components=n_pcs)
    pert_pca = pca.fit_transform(np.concatenate([background_pert, embeddings_pert]))

    # Concatenate embeddings after PCA
    embeddings_cell_all = np.concatenate([background_cell, embeddings_cell])
    embeddings_pert_all = pert_pca

    # Labels for plotting
    cat_pert = ["Other"] * background_pert.shape[0] + [highlight_label] * embeddings_pert.shape[0]
    cat_cell = ["Other"] * background_cell.shape[0] + [highlight_label] * embeddings_cell.shape[0]

    # Create UMAP transformers and transform data
    trans_pert = umap.UMAP(random_state=random_state, min_dist=0.5, n_neighbors=30).fit(embeddings_pert_all)
    trans_cell = umap.UMAP(random_state=random_state, min_dist=0.5, n_neighbors=30).fit(embeddings_cell_all)
    Y_embedded = trans_pert.transform(embeddings_pert_all)
    Z_embedded = trans_cell.transform(embeddings_cell_all)

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # Define a plotting function for each subplot
    def plot_with_contours(ax, data, categories, title, highlight, add_contour = True):
        highlight_data = data[categories == highlight]
        other_data = data[categories != highlight]

        # Plot background data
        ax.scatter(other_data[:, 0], other_data[:, 1], color='gray', s=1, label='Other')

        # Plot highlight data
        ax.scatter(highlight_data[:, 0], highlight_data[:, 1], color='red', s=1, label=highlight_label)

        if add_contour and highlight_data.size > 0:
            x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
            y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
            x_grid = np.linspace(x_min, x_max, 100)
            y_grid = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            kde = gaussian_kde(highlight_data.T, bw_method=bandwidth)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=5, colors='red')
        
        ax.set_xlim(np.min(data[:, 0]), np.max(data[:, 0]))
        ax.set_ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plotting for each representation
    plot_with_contours(ax1, Y_embedded, np.array(cat_pert), 'Perturbation Representation', highlight_label, add_contour = False)
    plot_with_contours(ax2, Z_embedded, np.array(cat_cell), 'Cellular Representation', highlight_label)

    # Draw lines for highlighted points
    transFigure = fig.transFigure.inverted()
    highlight_indices = np.where(np.array(cat_pert) == highlight_label)[0]
    if len(highlight_indices) > 30:
        highlight_indices = np.random.choice(highlight_indices, 30, replace=False)  # Randomly pick 30 indices

    for index in highlight_indices:
        xy1 = transFigure.transform(ax1.transData.transform(Y_embedded[index]))
        xy2 = transFigure.transform(ax2.transData.transform(Z_embedded[index]))
        line = matplotlib.lines.Line2D((xy1[0], xy2[0]), (xy1[1], xy2[1]), transform=fig.transFigure, color='red', linewidth=0.5)
        fig.lines.append(line)

    # Place a legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.show()