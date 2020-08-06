#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from itertools import cycle, islice

import scanpy as sc
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=200)

#Defining class that contains functions that will perform the mapping with XGBoost and plot the results
class TimeMapping():
    
    # xgbclassifier will run the feature selection, training and validation, and testing
    def xgbclassifier(
        self,
        train_anndata,
        test_anndata,
        train_dict,
        test_dict,
        max_cells_per_ident = 700,
        train_frac = 0.7
        ): 

        self.train_dict = train_dict
        self.test_dict = test_dict

        self.numbertrainclasses = len(train_anndata.obs.cluster.values.categories)
        self.numbertestclasses = len(test_anndata.obs.cluster.values.categories)

        #Splitting the cell barcodes into a training set and validation set based on the minimum of 70% of cells or 700 cells
        #Creating array of the labels for each cell (the cluster each cell barcode belongs too)
        training_set_train = []
        training_label_train = []

        for i in train_anndata.obs.cluster.values.categories.values:
            cells_in_clust = train_anndata.obs.index[train_anndata.obs.cluster.values == i]
            n = min(max_cells_per_ident,round(len(cells_in_clust)*train_frac))
            train_temp = np.random.choice(cells_in_clust,n,replace = False)
            if len(train_temp) < 100:
                train_temp_bootstrap = np.random.choice(train_temp, size = 100 - int(len(train_temp)))
                train_temp = np.hstack([train_temp_bootstrap, train_temp])
            training_set_train = np.hstack([training_set_train,train_temp])
            training_label_train = np.hstack([training_label_train,np.repeat(train_dict[i],len(train_temp))])

        training_set_test = []
        training_label_test = []

        for i in test_anndata.obs.cluster.values.categories.values:
            cells_in_clust = test_anndata.obs.index[test_anndata.obs.cluster.values == i]
            n = min(max_cells_per_ident,round(len(cells_in_clust)*train_frac))
            train_temp = np.random.choice(cells_in_clust,n,replace = False)
            if len(train_temp) < 100:
                train_temp_bootstrap = np.random.choice(train_temp, size = 100 - int(len(train_temp)))
                train_temp = np.hstack([train_temp_bootstrap, train_temp])
            training_set_test = np.hstack([training_set_test,train_temp])
            training_label_test = np.hstack([training_label_test,np.repeat(test_dict[i],len(train_temp))])

        train_index_train = []
        for i in training_set_train:
            train_index_train.append(np.where(train_anndata.obs.index.values == i)[0][0])

        train_index_test = []
        for i in training_set_test:
            train_index_test.append(np.where(test_anndata.obs.index.values == i)[0][0])

        train_matrix_train = xgb.DMatrix(data = train_anndata.raw.X.A[train_index_train,:], label = training_label_train, feature_names = train_anndata.var.index.values)

        train_matrix_test = xgb.DMatrix(data = test_anndata.raw.X.A[train_index_test,:], label = training_label_test, feature_names = test_anndata.var.index.values)

        del training_set_train, training_label_train, training_set_test, training_label_test, train_index_train, train_index_test

        #Defining parameters for the XGBoost Model
        xgb_params_train = {
            'objective':'multi:softprob',
            'eval_metric':'mlogloss',
            'num_class':self.numbertrainclasses,
            'eta':0.2,
            'max_depth':6,
            'subsample': 0.6}
        nround = 200

        #Fitting the XGBoost Model to the training data
        bst_model_train = xgb.train(
            params = xgb_params_train,
            dtrain = train_matrix_train,
            num_boost_round = nround)

        xgb_params_test = {
            'objective':'multi:softprob',
            'eval_metric':'mlogloss',
            'num_class':self.numbertestclasses,
            'eta':0.2,
            'max_depth':6,
            'subsample': 0.6}
        nround = 200

        #Fitting the XGBoost Model to the testing data
        bst_model_test = xgb.train(
            params = xgb_params_test,
            dtrain = train_matrix_test,
            num_boost_round = nround)

        train_xgboost_scores = bst_model_train.get_score(importance_type="gain")
        sort_train_scores = {k: v for k, v in sorted(train_xgboost_scores.items(), key=lambda item: item[1], reverse = True)[:500]}
        top500genestrain = list(sort_train_scores.keys())

        test_xgboost_scores = bst_model_test.get_score(importance_type="gain")
        sort_test_scores = {k: v for k, v in sorted(test_xgboost_scores.items(), key=lambda item: item[1], reverse = True)[:500]}
        top500genestest = list(sort_test_scores.keys())

        common_top_genes = np.array([i for i in top500genestrain if i in top500genestest]) #These are the features that we will use for training, validating and testing

        del train_matrix_train, train_matrix_test, bst_model_train, bst_model_test, train_xgboost_scores, sort_train_scores, top500genestrain, test_xgboost_scores, sort_test_scores, top500genestest

        #Train XGBoost on 70% of training data and validate on the remaining data
        common_top_genes_index_train = []
        for i in common_top_genes:
            common_top_genes_index_train.append(np.where(train_anndata.var.index.values == i)[0][0])

        training_set_train_70 = []
        validation_set_train_70 = []
        training_label_train_70 = []
        validation_label_train_70 = []

        for i in train_anndata.obs.cluster.values.categories.values:
            cells_in_clust = train_anndata.obs.index[train_anndata.obs.cluster.values == i]
            n = min(max_cells_per_ident,round(len(cells_in_clust)*train_frac))
            train_temp = np.random.choice(cells_in_clust,n,replace = False)
            validation_temp = np.setdiff1d(cells_in_clust, train_temp)
            if len(train_temp) < 100:
                train_temp_bootstrap = np.random.choice(train_temp, size = 100 - int(len(train_temp)))
                train_temp = np.hstack([train_temp_bootstrap, train_temp])
            training_set_train_70 = np.hstack([training_set_train_70,train_temp])
            validation_set_train_70 = np.hstack([validation_set_train_70,validation_temp])
            training_label_train_70 = np.hstack([training_label_train_70,np.repeat(train_dict[i],len(train_temp))])
            validation_label_train_70 = np.hstack([validation_label_train_70,np.repeat(train_dict[i],len(validation_temp))])

        train_index_train_70 = []
        for i in training_set_train_70:
            train_index_train_70.append(np.where(train_anndata.obs.index.values == i)[0][0])
        validation_index_train_70 = []
        for i in validation_set_train_70:
            validation_index_train_70.append(np.where(train_anndata.obs.index.values == i)[0][0])

        train_matrix_train_70 = xgb.DMatrix(data = train_anndata.raw.X.A[:,common_top_genes_index_train][train_index_train_70,:], label = training_label_train_70)
        validation_matrix_train_70 = xgb.DMatrix(data = train_anndata.raw.X.A[:,common_top_genes_index_train][validation_index_train_70,:], label = validation_label_train_70)

        del training_set_train_70, validation_set_train_70, training_label_train_70, train_index_train_70, validation_index_train_70

        bst_model_train_70 = xgb.train(
            params = xgb_params_train,
            dtrain = train_matrix_train_70,
            num_boost_round = nround)

        validation_pred_train_70 = bst_model_train_70.predict(data = validation_matrix_train_70)

        valid_predlabels_train_70 = np.zeros((validation_pred_train_70.shape[0]))
        for i in range(validation_pred_train_70.shape[0]):
            valid_predlabels_train_70[i] = np.argmax(validation_pred_train_70[i,:])
        
        del train_matrix_train_70, validation_matrix_train_70, validation_pred_train_70

        #Train XGBoost on the full training data
        training_set_train_full = []
        training_label_train_full = []

        for i in train_anndata.obs.cluster.values.categories.values:
            train_temp = train_anndata.obs.index[train_anndata.obs.cluster.values == i]
            if len(train_temp) < 100:
                train_temp_bootstrap = np.random.choice(train_temp, size = 100 - int(len(train_temp)))
                train_temp = np.hstack([train_temp_bootstrap, train_temp])
            training_set_train_full = np.hstack([training_set_train_full,train_temp])
            training_label_train_full = np.hstack([training_label_train_full,np.repeat(train_dict[i],len(train_temp))])

        train_index_full = []
        for i in training_set_train_full:
            train_index_full.append(np.where(train_anndata.obs.index.values == i)[0][0])

        full_training_data = xgb.DMatrix(data = train_anndata.raw.X.A[:,common_top_genes_index_train][train_index_full,:], label = training_label_train_full)

        del common_top_genes_index_train, training_set_train_full, training_label_train_full, train_index_full

        bst_model_full_train = xgb.train(
            params = xgb_params_train,
            dtrain = full_training_data,
            num_boost_round = nround)

        #Predict the testing cluster labels
        common_top_genes_index_test = []
        for i in common_top_genes:
            common_top_genes_index_test.append(np.where(test_anndata.var.index.values == i)[0][0])

        full_testing_data = xgb.DMatrix(data = test_anndata.raw.X.A[:,common_top_genes_index_test])
        test_prediction = bst_model_full_train.predict(data = full_testing_data)

        del bst_model_full_train, full_testing_data

        test_predlabels = np.zeros((test_prediction.shape[0]))
        for i in range(test_prediction.shape[0]):
            if np.max(test_prediction[i,:]) > 1.2*(1/self.numbertrainclasses):
                test_predlabels[i] = np.argmax(test_prediction[i,:])
            else:
                test_predlabels[i] = self.numbertrainclasses

        test_labels = np.zeros(len(test_anndata.obs.cluster.values))
        for i,l in enumerate(test_anndata.obs.cluster.values):
            test_labels[i] = test_dict[l]

        return validation_label_train_70, valid_predlabels_train_70, test_labels, test_predlabels

    #plotConfusionMatrix will take the results from the xgboost classifier and plot them
    def plotConfusionMatrix(
        self,
        ytrue,
        ypred,
        type,
        save_as,
        title = '',
        xaxislabel = '',
        yaxislabel = ''
        ):

        confusionmatrix = confusion_matrix(y_true = ytrue, y_pred = ypred)
        if type == 'mapping':
          if self.numbertrainclasses in ypred:
            confusionmatrix = confusionmatrix[0:self.numbertestclasses,0:self.numbertrainclasses+1]
          else:
            confusionmatrix = confusionmatrix[0:self.numbertestclasses,0:self.numbertrainclasses]
        confmatpercent = np.zeros(confusionmatrix.shape)
        for i in range(confusionmatrix.shape[0]):
          if np.sum(confusionmatrix[i,:]) != 0:
            confmatpercent[i,:] = confusionmatrix[i,:]/np.sum(confusionmatrix[i,:])
          else:
            confmatpercent[i,:] = confusionmatrix[i,:]
        diagcm = confmatpercent
        xticks = np.linspace(0, confmatpercent.shape[1]-1, confmatpercent.shape[1], dtype = int)
        xticksactual = []
        for i in xticks:
          if i != self.numbertrainclasses:
            xticksactual.append(list(self.train_dict.keys())[i])
          else:
            xticksactual.append('Unassigned')
        dot_max = np.max(diagcm.flatten())
        dot_min = 0
        if dot_min != 0 or dot_max != 1:
            frac = np.clip(diagcm, dot_min, dot_max)
            old_range = dot_max - dot_min
            frac = (frac - dot_min) / old_range
        else:
            frac = diagcm
        xvalues = []
        yvalues = []
        sizes = []
        for i in range(diagcm.shape[0]):
            for j in range(diagcm.shape[1]):
                xvalues.append(j)
                yvalues.append(i)
                sizes.append((frac[i,j]*35)**1.5)
        size_legend_width = 0.5
        height = diagcm.shape[0] * 0.3 + 1
        height = max([1.5, height])
        heatmap_width = diagcm.shape[1] * 0.35
        width = (
            heatmap_width
            + size_legend_width
            )
        fig = plt.figure(figsize=(width, height))
        axs = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            wspace=0.02,
            hspace=0.04,
            width_ratios=[
                        heatmap_width,
                        size_legend_width
                        ],
            height_ratios = [0.5, 10]
            )
        dot_ax = fig.add_subplot(axs[1, 0])
        dot_ax.scatter(xvalues,yvalues, s = sizes, c = 'blue', norm=None, edgecolor='none')
        y_ticks = range(diagcm.shape[0])
        dot_ax.set_yticks(y_ticks)
        if type == 'validation':
          dot_ax.set_yticklabels(list(self.train_dict.keys()))
        elif type == 'mapping':
          dot_ax.set_yticklabels(list(self.test_dict.keys()))
        x_ticks = range(diagcm.shape[1])
        dot_ax.set_xticks(x_ticks)
        dot_ax.set_xticklabels(xticksactual, rotation=90)
        dot_ax.tick_params(axis='both', labelsize='small')
        dot_ax.grid(True, linewidth = 0.2)
        dot_ax.set_axisbelow(True)
        dot_ax.set_xlim(-0.5, diagcm.shape[1] + 0.5)
        ymin, ymax = dot_ax.get_ylim()
        dot_ax.set_ylim(ymax + 0.5, ymin - 0.5)
        dot_ax.set_xlim(-1, diagcm.shape[1])
        dot_ax.set_xlabel(xaxislabel)
        dot_ax.set_ylabel(yaxislabel)
        dot_ax.set_title(title)
        size_legend_height = min(1.75, height)
        wspace = 10.5 / width
        axs3 = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=axs[1, 1],
            wspace=wspace,
            height_ratios=[
                        size_legend_height / height,
                        (height - size_legend_height) / height
                        ]
            )
        diff = dot_max - dot_min
        if 0.3 < diff <= 0.6:
            step = 0.1
        elif diff <= 0.3:
            step = 0.05
        else:
            step = 0.2
        fracs_legends = np.arange(dot_max, dot_min, step * -1)[::-1]
        if dot_min != 0 or dot_max != 1:
            fracs_values = (fracs_legends - dot_min) / old_range
        else:
            fracs_values = fracs_legends
        size = (fracs_values * 35) ** 1.5
        size_legend = fig.add_subplot(axs3[0])
        size_legend.scatter(np.repeat(0, len(size)), range(len(size)), s=size, c = 'blue')
        size_legend.set_yticks(range(len(size)))
        labels = ["{:.0%}".format(x) for x in fracs_legends]
        if dot_max < 1:
            labels[-1] = ">" + labels[-1]
        size_legend.set_yticklabels(labels)
        size_legend.set_yticklabels(["{:.0%}".format(x) for x in fracs_legends])
        size_legend.tick_params(axis='y', left=False, labelleft=False, labelright=True)
        size_legend.tick_params(axis='x', bottom=False, labelbottom=False)
        size_legend.spines['right'].set_visible(False)
        size_legend.spines['top'].set_visible(False)
        size_legend.spines['left'].set_visible(False)
        size_legend.spines['bottom'].set_visible(False)
        size_legend.grid(False)
        ymin, ymax = size_legend.get_ylim()
        size_legend.set_ylim(ymin, ymax + 0.5)
        fig.savefig(save_as, bbox_inches = 'tight')

        return diagcm, xticks, axs

#Loading in the corresponding h5ad files
P8_adata = sc.read_h5ad('P8_glut.h5ad')
P14_adata = sc.read_h5ad('P14_glut.h5ad')
P17_adata = sc.read_h5ad('P17_glut.h5ad')
P21_adata = sc.read_h5ad('P21_glut.h5ad')
P28_adata = sc.read_h5ad('P28_glut.h5ad')
P38_adata = sc.read_h5ad('P38_glut.h5ad')

#Assigning integer values to each cluster at each age in order to be able to use XGBoost
P8_dict = {'L2/3_A':0, 'L2/3_B':1, 'L2/3_C1':2, 'L2/3_C2':3, 'L4_AB':4, 'L4_C':5, 'L5IT':6, 'L5NP':7, 'L6IT_A':8, 'L6IT_B':9, 'L6CT_A':10, 'L6CT_B':11, 'L6CT_C':12, 'L6b':13}
P14_dict = {'L2/3_A':0, 'L2/3_B':1, 'L2/3_C':2, 'L2/3_Ambig':3, 'L4_A':4, 'L4_B':5, 'L4_C':6, 'L5IT':7, 'L5NP':8, 'L5PT_A':9, 'L5PT_B':10, 'L6IT_A':11, 'L6IT_B':12, 'L6CT_A':13, 'L6CT_B':14, 'L6CT_C':15, 'L6b':16}
P17_dict = {'L2/3_A':0, 'L2/3_B':1, 'L2/3_C':2, 'L4_A':3, 'L4_B':4, 'L4_C':5, 'L5IT':6, 'L5NP':7, 'L5PT_A':8, 'L5PT_B':9, 'L6IT_A':10, 'L6IT_B':11, 'L6CT_A':12, 'L6CT_B':13, 'L6CT_C':14, 'L6b':15}
P21_dict = {'L2/3_A':0, 'L2/3_B':1, 'L2/3_C':2, 'L4_A':3, 'L4_B':4, 'L4_C':5, 'L5IT':6, 'L5NP':7, 'L5PT_A':8, 'L5PT_B':9, 'L6IT_A':10, 'L6IT_B':11, 'L6CT_A':12, 'L6CT_B':13, 'L6CT_C':14, 'L6b':15}
P28_dict = {'L2/3_A':0, 'L2/3_B':1, 'L2/3_C':2, 'L4_A':3, 'L4_B':4, 'L4_C1':5, 'L4_C2':6, 'L5IT':7, 'L5NP':8, 'L5PT_A':9, 'L5PT_B':10, 'L6IT_A':11, 'L6IT_B':12, 'L6CT_A':13, 'L6CT_B':14, 'L6CT_C':15, 'L6b':16}
P38_dict = {'L2/3_A':0, 'L2/3_B':1, 'L2/3_C':2, 'L4_A':3, 'L4_B':4, 'L4_C':5, 'L5IT':6, 'L5NP':7, 'L5PT_A':8, 'L5PT_B':9, 'L6IT_A':10, 'L6IT_B':11, 'L6CT_A':12, 'L6CT_B':13, 'L6CT_C':14, 'L6b':15}

#Run the mapping for P8 to P14
tm = TimeMapping()
validation_label_train_70P8vsP14, valid_predlabels_train_70P8vsP14, test_labelsP8vsP14, test_predlabelsP8vsP14 = tm.xgbclassifier(
    train_anndata = P8_adata,
    test_anndata = P14_adata,
    train_dict = P8_dict,
    test_dict = P14_dict
    )
#Plot and save the validation confusion matrix
validationconfmatP8vsP14, validationxticksP8vsP14, validationplotP8vsP14 = tm.plotConfusionMatrix(
    ytrue = validation_label_train_70P8vsP14,
    ypred = valid_predlabels_train_70P8vsP14,
    type = 'validation',
    save_as = 'GlutP8_P14Validation.pdf',
    title = '',
    xaxislabel = 'Predicted',
    yaxislabel = 'True'
    )
#Plot and save the mapping confusion matrix
mappingconfmatP8vsP14, mappingxticksP8vsP14, mappingplotP8vsP14 = tm.plotConfusionMatrix(
    ytrue = test_labelsP8vsP14,
    ypred = test_predlabelsP8vsP14,
    type = 'mapping',
    save_as = 'GlutP8_P14Mapping.pdf',
    title = '',
    xaxislabel = 'P8',
    yaxislabel = 'P14'
    )
del tm

#Repeat for all other mappings
tm = TimeMapping()
validation_label_train_70P14vsP17, valid_predlabels_train_70P14vsP17, test_labelsP14vsP17, test_predlabelsP14vsP17 = tm.xgbclassifier(
    train_anndata = P14_adata,
    test_anndata = P17_adata,
    train_dict = P14_dict,
    test_dict = P17_dict
    )
validationconfmatP14vsP17, validationxticksP14vsP17, validationplotP14vsP17 = tm.plotConfusionMatrix(
    ytrue = validation_label_train_70P14vsP17,
    ypred = valid_predlabels_train_70P14vsP17,
    type = 'validation',
    save_as = 'GlutP14_P17Validation.pdf',
    title = '',
    xaxislabel = 'Predicted',
    yaxislabel = 'True'
    )
mappingconfmatP14vsP17, mappingxticksP14vsP17, mappingplotP14vsP17 = tm.plotConfusionMatrix(
    ytrue = test_labelsP14vsP17,
    ypred = test_predlabelsP14vsP17,
    type = 'mapping',
    save_as = 'GlutP14_P17Mapping.pdf',
    title = '',
    xaxislabel = 'P14',
    yaxislabel = 'P17'
    )
del tm

tm = TimeMapping()
validation_label_train_70P17vsP21, valid_predlabels_train_70P17vsP21, test_labelsP17vsP21, test_predlabelsP17vsP21 = tm.xgbclassifier(
    train_anndata = P17_adata,
    test_anndata = P21_adata,
    train_dict = P17_dict,
    test_dict = P21_dict
    )
validationconfmatP17vsP21, validationxticksP17vsP21, validationplotP17vsP21 = tm.plotConfusionMatrix(
    ytrue = validation_label_train_70P17vsP21,
    ypred = valid_predlabels_train_70P17vsP21,
    type = 'validation',
    save_as = 'GlutP17_P21Validation.pdf',
    title = '',
    xaxislabel = 'Predicted',
    yaxislabel = 'True'
    )
mappingconfmatP17vsP21, mappingxticksP17vsP21, mappingplotP17vsP21 = tm.plotConfusionMatrix(
    ytrue = test_labelsP17vsP21,
    ypred = test_predlabelsP17vsP21,
    type = 'mapping',
    save_as = 'GlutP17_P21Mapping.pdf',
    title = '',
    xaxislabel = 'P17',
    yaxislabel = 'P21'
    )
del tm

tm = TimeMapping()
validation_label_train_70P21vsP28, valid_predlabels_train_70P21vsP28, test_labelsP21vsP28, test_predlabelsP21vsP28 = tm.xgbclassifier(
    train_anndata = P21_adata,
    test_anndata = P28_adata,
    train_dict = P21_dict,
    test_dict = P28_dict
    )
validationconfmatP21vsP28, validationxticksP21vsP28, validationplotP21vsP28 = tm.plotConfusionMatrix(
    ytrue = validation_label_train_70P21vsP28,
    ypred = valid_predlabels_train_70P21vsP28,
    type = 'validation',
    save_as = 'GlutP21_P28Validation.pdf',
    title = '',
    xaxislabel = 'Predicted',
    yaxislabel = 'True'
    )
mappingconfmatP21vsP28, mappingxticksP21vsP28, mappingplotP21vsP28 = tm.plotConfusionMatrix(
    ytrue = test_labelsP21vsP28,
    ypred = test_predlabelsP21vsP28,
    type = 'mapping',
    save_as = 'GlutP21_P28Mapping.pdf',
    title = '',
    xaxislabel = 'P21',
    yaxislabel = 'P28'
    )
del tm

tm = TimeMapping()
validation_label_train_70P28vsP38, valid_predlabels_train_70P28vsP38, test_labelsP28vsP38, test_predlabelsP28vsP38 = tm.xgbclassifier(
    train_anndata = P28_adata,
    test_anndata = P38_adata,
    train_dict = P28_dict,
    test_dict = P38_dict
    )
validationconfmatP28vsP38, validationxticksP28vsP38, validationplotP28vsP38 = tm.plotConfusionMatrix(
    ytrue = validation_label_train_70P28vsP38,
    ypred = valid_predlabels_train_70P28vsP38,
    type = 'validation',
    save_as = 'GlutP28_P38Validation.pdf',
    title = '',
    xaxislabel = 'Predicted',
    yaxislabel = 'True'
    )
mappingconfmatP28vsP38, mappingxticksP28vsP38, mappingplotP28vsP38 = tm.plotConfusionMatrix(
    ytrue = test_labelsP28vsP38,
    ypred = test_predlabelsP28vsP38,
    type = 'mapping',
    save_as = 'GlutP28_P38Mapping.pdf',
    title = '',
    xaxislabel = 'P28',
    yaxislabel = 'P38'
    )

#Creating the Sankey Plot
#Order of the cell types for each age on the Sankey plot
P8_labels = ['L2/3_A', 'L2/3_B', 'L2/3_C1', 'L2/3_C2', 'L4_AB', 'L4_C', 'L5IT', 'L5NP', 'L6IT_A', 'L6IT_B', 'L6CT_A', 'L6CT_B', 'L6CT_C', 'L6b']
P14_labels = ['L2/3_A', 'L2/3_B', 'L2/3_C', 'L2/3_Ambig', 'L4_A', 'L4_B', 'L4_C', 'L5IT', 'L5NP', 'L5PT_A', 'L5PT_B', 'L6IT_A', 'L6IT_B', 'L6CT_A', 'L6CT_B', 'L6CT_C', 'L6b']
P17_labels = ['L2/3_A', 'L2/3_B', 'L2/3_C', 'L4_A', 'L4_B', 'L4_C', 'L5IT', 'L5NP', 'L5PT_A', 'L5PT_B', 'L6IT_A', 'L6IT_B', 'L6CT_A', 'L6CT_B', 'L6CT_C', 'L6b']
P21_labels = ['L2/3_A', 'L2/3_B', 'L2/3_C', 'L4_A', 'L4_B', 'L4_C', 'L5IT', 'L5NP', 'L5PT_A', 'L5PT_B', 'L6IT_A', 'L6IT_B', 'L6CT_A', 'L6CT_B', 'L6CT_C', 'L6b']
P28_labels = ['L2/3_A', 'L2/3_B', 'L2/3_C', 'L4_A', 'L4_B', 'L4_C1', 'L4_C2', 'L5IT', 'L5NP', 'L5PT_A', 'L5PT_B', 'L6IT_A', 'L6IT_B', 'L6CT_A', 'L6CT_B', 'L6CT_C', 'L6b']
P38_labels = ['L2/3_A', 'L2/3_B', 'L2/3_C', 'L4_A', 'L4_B', 'L4_C', 'L5IT', 'L5NP', 'L5PT_A', 'L5PT_B', 'L6IT_A', 'L6IT_B', 'L6CT_A', 'L6CT_B', 'L6CT_C', 'L6b']

#Defining the source, target and value in order to specify the Sankey plot
sources = []
targets = []
values = []
lst = list(np.linspace(0,len(P8_labels)-1,len(P8_labels)))
sources.extend(list(islice(cycle(lst), len(P8_labels)*len(P14_labels))))
targets.extend(list(np.repeat(np.linspace(0,len(P14_labels)-1,len(P14_labels)), len(P8_labels)) + len(P8_labels)))
values.extend(list(mappingconfmatP8vsP14.flatten()))
lst = list(np.linspace(0,len(P14_labels)-1,len(P14_labels)) + len(P8_labels))
sources.extend(list(islice(cycle(lst), len(P14_labels)*len(P17_labels))))
targets.extend(list(np.repeat(np.linspace(0,len(P17_labels)-1,len(P17_labels)), len(P14_labels)) + len(P8_labels) + len(P14_labels)))
values.extend(list(mappingconfmatP14vsP17.flatten()))
lst = list(np.linspace(0,len(P17_labels)-1,len(P17_labels)) + len(P8_labels) + len(P14_labels))
sources.extend(list(islice(cycle(lst), len(P17_labels)*len(P21_labels))))
targets.extend(list(np.repeat(np.linspace(0,len(P21_labels)-1,len(P21_labels)), len(P17_labels)) + len(P8_labels) + len(P14_labels)+ len(P17_labels)))
values.extend(list(mappingconfmatP17vsP21.flatten()))
lst = list(np.linspace(0,len(P21_labels)-1,len(P21_labels)) + len(P8_labels) + len(P14_labels) + len(P17_labels))
sources.extend(list(islice(cycle(lst), len(P21_labels)*len(P28_labels))))
targets.extend(list(np.repeat(np.linspace(0,len(P28_labels)-1,len(P28_labels)), len(P21_labels)) + len(P8_labels) + len(P14_labels)+ len(P17_labels) + len(P21_labels)))
values.extend(list(mappingconfmatP21vsP28.flatten()))
lst = list(np.linspace(0,len(P28_labels)-1,len(P28_labels)) + len(P8_labels) + len(P14_labels) + len(P17_labels) + len(P21_labels))
sources.extend(list(islice(cycle(lst), len(P28_labels)*len(P38_labels))))
targets.extend(list(np.repeat(np.linspace(0,len(P38_labels)-1,len(P38_labels)), len(P28_labels)) + len(P8_labels) + len(P14_labels)+ len(P17_labels) + len(P21_labels) + len(P28_labels)))
values.extend(list(mappingconfmatP28vsP38.flatten()))

#Defining the positions of the nodes for the Sankey plot
yspacing = list(np.linspace(0.01,1.2,len(P14_labels)))
P8_x = list(np.repeat(0.01,len(P8_labels)))
P8_y = list(np.linspace(yspacing[1],yspacing[-2],len(P8_labels)))
P14_x = list(np.repeat(0.01 + 0.17,len(P14_labels)))
P14_y = list(np.linspace(0.01,1.2,len(P14_labels)))
P17_x = list(np.repeat(0.01 + (2*0.17),len(P17_labels)))
P17_y = list(np.linspace(0.02,1.19,len(P17_labels)))
P21_x = list(np.repeat(0.01 + (3*0.17),len(P17_labels)))
P21_y = list(np.linspace(0.02,1.19,len(P17_labels)))
P28_x = list(np.repeat(0.01 + (4*0.17),len(P28_labels)))
P28_y = list(np.linspace(0.01,1.2,len(P28_labels)))
P38_x = list(np.repeat(0.01 + (5*0.17),len(P38_labels)))
P38_y = list(np.linspace(0.02,1.19,len(P38_labels)))

#Assigning colors to the links between the nodes based on the value
linecolor = []
for i in values:
  if i > 0.8:
    linecolor.append('rgba(0.0, 0.0, 1.0, 0.75)')
  elif 0.5 < i <= 0.8:
    linecolor.append('rgba(0.5411764705882353, 0.16862745098039217, 0.8862745098039215, 0.55)')
  elif 0.2 < i <= 0.5:
    linecolor.append('rgba(0.4392156862745098, 0.5019607843137255, 0.5647058823529412, 0.35)')
  else:
    linecolor.append('rgba(0.6862745098039216, 0.9333333333333333, 0.9333333333333333, 0.25)')

#Assigning colors to the nodes based on cell class
colorarray = ['maroon', 'maroon', 'maroon', 'maroon', 'maroon', 'maroon',
              'gold', 'gold', 'gold', 'gold', 'gold', 'gold',
              'darkorange', 'darkorange', 'darkorange', 'darkorange',
              'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen']
all_labels = np.array(P8_labels + P14_labels + P17_labels + P21_labels + P28_labels + P38_labels)
unique_labels = np.unique(all_labels)
colors = list(np.zeros(all_labels.shape, dtype = str))
for i, label in enumerate(unique_labels):
  index = np.where(all_labels == label)[0]
  colorvalue = colorarray[i]
  for j in index:
    colors[j] = colorvalue

#Creating the Sankey plot
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 50,
      thickness = 30,
      line = dict(color = "black", width = 0.5),
      label = P8_labels + P14_labels + P17_labels + P21_labels + P28_labels + P38_labels,
      color = colors,
      x = P8_x + P14_x + P17_x + P21_x + P28_x + P38_x,
      y = P8_y + P14_y + P17_y + P21_y + P28_y + P38_y
    ),
    link = dict(
      source = sources,
      target = targets,
      value = values,
      color = linecolor
  ))])

fig.show()

#Create the legend for the Sankey plot
fig, axes = plt.subplots(1,1, figsize = (2.5,3))
size = np.linspace(1,4,4)
axes.scatter(np.repeat(0, len(size)), range(0,len(size)), marker = 's', s = 900, c = ['#afeeee40', '#70809059', '#8a2be28c', '#0000ffbf'])
axes.set_yticks(range(len(size)))
labels = [r'$\leq 0.2$', r'$0.2 < x \leq 0.5$', r'$0.5 < x \leq 0.8$', r'$> 0.8$']
axes.set_yticklabels(labels)
axes.tick_params(axis='y', left=False, labelleft=False, labelright=True)
axes.tick_params(axis='x', bottom=False, labelbottom=False)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.grid(False)
ymin, ymax = axes.get_ylim()
axes.set_ylim(ymin - 0.5, ymax + 0.5)
axes.tick_params(axis='both', labelsize=15)
fig.tight_layout()
fig.savefig('glut_sankey_legend.pdf')