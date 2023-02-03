#This file calculates the mean ECE and ACE over all tasks.

import sparsechem as sc
import numpy as np

def calculateErrors_allTargets(y_true, y_score, num_bins=10):
    '''
    Args:
    y_true: matrix of observations in csr format ()
    y_score: matrix of probability-like predictions in csr format
    y_bin: number of bins to which datapoints are assigned
    
    Returns a list of the Expected Calibration Error (ECEperTarget) and the Adaptive Calbration Error (ACEperTarget) for every target for a classificatino task.
    
    '''        
    #split input array according to predictions in equally spaced bins
    def split_arrays_ECE(true_labels, predictions, bounds, index):
        true_labels_split=true_labels[np.logical_and(predictions>bounds[index], predictions<=bounds[index+1])].T
        predictions_split=predictions[np.logical_and(predictions>bounds[index], predictions<=bounds[index+1])].T
        bin_size=true_labels_split.shape[0]
        return true_labels_split, predictions_split, bin_size
    
    #split input array according to predictions in equally sized bins
    def split_arrays_ACE(true_labels, predictions, nr_bins, index):
        true_labels_split=np.array_split(true_labels, nr_bins)[index]
        predictions_split=np.array_split(predictions, nr_bins)[index]
        bin_size=true_labels_split.shape[0]
        return true_labels_split, predictions_split, bin_size

    #Calculate positive ratio (=accuracy)
    #if there are no measurements (=no predictions) in a split: add 0 to acc list
    #Note: if 0 is added to the list, the difference between acc and conf is the conf of this split
    def calculate_posRatio(input_arr):
        return (input_arr==1).sum()/input_arr.shape[0]

    #Calculate Mean of Probablities in Column (=confidence)
    #if there are no measurements(=no predictions) in a split: the confidence is calculated from the bounds_ECE list
    def calculate_probMean(input_arr):
        return np.mean(input_arr)
        
    #calculate Calibration Error (Nixon et al.: https://doi.org/10.48550/arXiv.1904.01685):
    def calculate_error(posRatio_bin, meanProb_bin, bin_size):
        return (np.abs(np.array(posRatio_bin)-np.array(meanProb_bin))*bin_size)
            #      |               acc(b)-         conf(n)| * nb
            
    def main():
        
        ECEperTarget=[]
        ACEperTarget=[]
        target_ind=0

        #bounds for bins used for ECE caluculation
        bounds_ECE=list(np.linspace(0,1,num_bins+1))
        
        #calculate ECE/ACE for each target  
        for target_ind in range(y_score.shape[1]):
                          
            #for ECE: specify Target and select nonzero entries
            nonzero_entries=y_true[:, target_ind].nonzero()[0]
            y_score_target=y_score[:, target_ind][nonzero_entries]
            y_true_target=y_true[:, target_ind][nonzero_entries]
            target_size=y_true_target.shape[0]
            
            #if the target has no measurements (target_size=0), the error of this target is nan.
            if target_size==0:
                ECEperTarget.append(np.nan)
                ACEperTarget.append(np.nan)
                continue
            
            #if there are measurements for the target (target_size>0): calculate ECE/ACE
                       
            #for ACE: sort y_true and y_score file by ascending probablity values in y_score file:
            index_sort_y_score=np.argsort(y_score_target, axis=0).A.flatten()
            y_score_sorted=y_score_target[index_sort_y_score]
            y_true_sorted=y_true_target[index_sort_y_score]    
            
            bin_ind=0
            ece=0
            ace=0

            #iterate over all bins
            for bin_ind in range(num_bins):

                #for current bin...
                #...split dataset and select split
                
                ECE_collector=split_arrays_ECE(y_true_target, y_score_target, bounds_ECE, bin_ind)
                ACE_collector=split_arrays_ACE(y_true_sorted, y_score_sorted, num_bins, bin_ind)
                
                if ECE_collector[2]!=0:
                    #...obtain positive ratio (=acc calculated from true values) for current bin if bin has measurements
                    posRatio_ECE=calculate_posRatio(ECE_collector[0])
                    #...obtain probablity mean (=conf calculated from predictions) for each split for current bin if bin has measurements
                    meanProb_ECE=calculate_probMean(ECE_collector[1])
                    #...calculate ECE for current target and sum error over all bins:
                    ece += calculate_error(posRatio_bin=posRatio_ECE, meanProb_bin=meanProb_ECE, bin_size=ECE_collector[2])
                
                
                
                if ACE_collector[2]!=0:
                    #...obtain positive ratio (=acc calculated from true values) for current bin if bin has measurements
                    posRatio_ACE=calculate_posRatio(ACE_collector[0])
                    #...obtain probablity mean (=conf calculated from predictions) for each split for current bin if bin has measurements
                    meanProb_ACE=calculate_probMean(ACE_collector[1])
                    #...calculate ACE for current bin and sum error over all bins qith measurements:
                    ace += calculate_error(posRatio_bin=posRatio_ACE, meanProb_bin=meanProb_ACE, bin_size=ACE_collector[2])
                

            #the final ECE/ACE is divided by number of datapoints in a target (ECE)/ by the number of bins (ACE)
            ece_target=ece/target_size
            ace_target=ace/target_size
            
            #Append ECE to list of ECEs/ACEs (one ECE/ACE per target)
            ECEperTarget.append(ece_target)
            ACEperTarget.append(ace_target)
            

        #print('ACE per Target', ACEperTarget)
        return ECEperTarget, ACEperTarget

    y_true=y_true.todense()
    y_score=y_score.todense()
    return main()


'''
Note:
If there are targets without measurements, 'nan' is added to the list of ECEs/ACEs.

'''



