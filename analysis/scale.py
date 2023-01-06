import cloudpickle
import pickle
import gzip
import os
from collections import defaultdict, OrderedDict
from coffea import hist, processor 
from coffea.util import load, save



def scale_file(file):

    print('Loading file:',file)    
    hists=load(file)
    #print('hists sumw', hists['sumw'].identifiers('dataset'))
    print(hists.keys())

    pd = []
    for d in hists['sumw'].identifiers('dataset'):
        print('sumw dataset', d)
        dataset = d.name
        if dataset.split("____")[0] not in pd: pd.append(dataset.split("____")[0])
    print('List of primary datasets:',pd)

    ##
    # Aggregate all the histograms that belong to a single dataset
    ##

    dataset = hist.Cat("dataset", "dataset", sorting='placement')
    dataset_cats = ("dataset",)
    dataset_map = OrderedDict()
    for pdi in pd:
        dataset_map[pdi] = (pdi+"*",)
    for key in hists.keys():
        hists[key] = hists[key].group(dataset_cats, dataset, dataset_map)
    print('Datasets aggregated')

    return scale(hists)

def scale_directory(directory):

    hists = {}
    for filename in os.listdir(directory):
        if '.merged' not in filename: continue
        print('Opening:', filename)
        hin = load(directory+'/'+filename)
        hists.update(hin)

    return scale(hists)

def scale(hists):

    ###
    # Rescaling MC histograms using the xsec weight
    ###
    print('print identifiers',hists['dibjetmass'].identifiers('dataset'))

    scale={}
    for d in hists['sumw'].identifiers('dataset'):
        scale[d]=hists['sumw'].integrate('dataset', d).values(overflow='all')[()][1]
        print('d: ', d)
        print(hists['sumw'].integrate('dataset', d).values(overflow='all')[()][1])
        print(hists['sumw'].integrate('dataset', d).values(overflow='all')[()][0])
    print('Sumw extracted')

    for key in hists.keys():
        if key=='sumw': continue
        for d in hists[key].identifiers('dataset'):
            print('dataset d', d)
            if 'BT' in d.name or 'MET' in d.name or 'SingleElectron' in d.name or 'SinglePhoton' in d.name or 'EGamma' in d.name or 'SingleMuon' in d.name: continue
            hists[key].scale({d:1/scale[d]},axis='dataset')
            print('scale[',d,']: ', scale[d])
    print('Histograms scaled')
    
    # printing the lumi DF
    for k,v in scale.items():
        print('dataset:{} sumgenweights:{}'.format(k,v))

    ###
    # Defining 'process', to aggregate different samples into a single process
    ##

    process = hist.Cat("process", "Process", sorting='placement')
    cats = ("dataset",)
    sig_map = OrderedDict()
    bkg_map = OrderedDict()
    data_map = OrderedDict()
    

    
    
    bkg_map["QCD"] = ("*QCD*",)
    bkg_map["TTJet"] = ("*TT*",)
    sig_map["ZprimeTobb200_dbs0p04"] = ("*dbs0p04*") #("*ZprimeTobb200_dbs0p04*",)  ## signals
    sig_map["ZprimeTobb200_dbs0p50"] = ("*dbs0p50*") #("*ZprimeTobb200_dbs0p50*",)
    sig_map["ZprimeTobb200_dbs1p00"] = ("*dbs1p00*") #("*ZprimeTobb200_dbs1p00*",)
    #     sig_map["MonoJet"] = ("MonoJet*",)  ## signals
    #     sig_map["MonoW"] = ("MonoW*",)    ## signals
    #sig_map["MonoZ"] = ("MonoZ*",)    ## signals
    data_map["data"] = ("*BT*", )
    #    data_map["SingleElectron"] = ("SingleElectron*", )
    #    data_map["SinglePhoton"] = ("SinglePhoton*", )
    #    data_map["SingleMuon"] = ("SingleMuon*", )
    #    data_map["EGamma"] = ("EGamma*", )
    #    data_map["Data"] = (["SingleElectron","SinglePhoton"], )
    print('Processes defined')
    print('bkg_map: ', bkg_map)
    
    ###
    # Storing signal and background histograms
    ###
    bkg_hists={}
    sig_hists={}
    data_hists={}
    for key in hists.keys():
        bkg_hists[key] = hists[key].group(cats, process, bkg_map)
        sig_hists[key] = hists[key].group(cats, process, sig_map)
        data_hists[key] = hists[key].group(cats, process, data_map)
    print('Histograms grouped')

    return bkg_hists, sig_hists, data_hists

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-f', '--file', help='file', dest='file')
    parser.add_option('-d', '--directory', help='directory', dest='directory')
    (options, args) = parser.parse_args()
    print('file option', options.file)

    if options.directory: 
        bkg_hists, sig_hists, data_hists = scale_directory(options.directory)
        name = options.directory
    if options.file: 
        bkg_hists, sig_hists, data_hists = scale_file(options.file)
        name = options.file.split(".")[0]

    hists={
        'bkg': bkg_hists,
        'sig': sig_hists,
        'data': data_hists
    }
    save(hists,name+'.scaled')
