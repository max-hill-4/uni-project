
params = {
        "b_size" : 16,
        "filter_size" : 5,
        "iterations" : 50,
        "k_folds" : 3,
        "in_channels" : 1
}

param_options = {
    "feature_freq" : 
        [[{'coh' : 'delta'}],
        [{'coh' : 'alpha'}]
        [{'coh' : 'delta'}, {'coh' : 'beta'}], 
        [{'coh' : 'alpha'}, {'coh' : 'theta'}],],
    "hormones" : [  
        ['BDC1'],
        ['BDC1.1'], ['BDC1.2'], ['BDC1.3'], ['BDC1.4'],
        ['BDC1.5'], ['BDC1.6'], ['BDC1.7'], ['BDC1.8'], 
        ['BDC1.9'],['BDC1.10'], ['BDC1.11']],
       

    "sleep_stages" : [['N1', 'N2', 'N3', 'REM']],
}
