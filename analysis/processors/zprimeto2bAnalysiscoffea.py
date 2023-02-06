#!/usr/bin/env python

import time
from coffea import hist, nanoevents, util
from coffea.util import load, save
import coffea.processor as processor
import awkward as ak
import numpy as np
import glob as glob
import re
import itertools
# import vector as vec
from coffea.nanoevents.methods import vector, candidate
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea.lumi_tools import LumiMask
# for applying JECs
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor
# from jmeCorrections import JetMetCorrections
import json


import coffea.processor as processor
from coffea import hist
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod

NanoAODSchema.warn_missing_crossrefs = False
from optparse import OptionParser
import pickle
np.errstate(invalid='ignore', divide='ignore')

class AnalysisProcessor(processor.ProcessorABC):

    lumis = {  # Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/TWikiLUM
        '2016': 19.52, #preVFP
        #'2016': 16.81, #postVFP
        '2017': 41.48,
        '2018': 59.83
    }

    met_filter_flags = {
        
        '2016': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter'
                 ],

        '2017': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'BadPFMuonDzFilter',
                 'eeBadScFilter',
                 'ecalBadCalibFilter'
                 ],

        '2018': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'BadPFMuonDzFilter',
                 'eeBadScFilter',
                 'ecalBadCalibFilter'
                 ],

    }

    def __init__(self, year, xsec, corrections, ids, common):

        self._year = year

        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])

        self._xsec = xsec
        
        self._samples = {
            'sr':('TT','QCD', 'Zprime'),
        }

        self._jet_triggers = {
            '2016': [
                'DoublePFJets40_CaloBTagCSV'
            ],
            '2017':[
                'DoublePFJets40_CaloBTagCSV',
                'PFHT180'
                
            ],
            '2018':[
                'DoublePFJets40_CaloBTagCSV',
                'PFHT180'
            ]
        }
        self._jet_triggers_sr = {
            '2016': [
                'DoublePFJets40_CaloBTagCSV'
            ],
            '2017':[
                'DoublePFJets40_CaloBTagCSV',
                'PFHT180'
                
            ],
            '2018':[
                'DoublePFJets40_CaloBTagCSV',
                'PFHT180'
            ]
        }
        self._jec = {
            
            '2016':  
            {
                'no_apv':
                ['Summer19UL16_V7_MC_L1FastJet_AK4PFchs',
                 'Summer19UL16_V7_MC_L2L3Residual_AK4PFchs',
                 'Summer19UL16_V7_MC_L2Relative_AK4PFchs',
                 'Summer19UL16_V7_MC_L2Residual_AK4PFchs',
                 'Summer19UL16_V7_MC_L3Absolute_AK4PFchs',
                 
             ],
                
                'apv':
                ['Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs',
                 'Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs',
                 'Summer19UL16APV_V7_MC_L2Relative_AK4PFchs',
                 'Summer19UL16APV_V7_MC_L2Residual_AK4PFchs',
                 'Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs',
                 
             ]
            },
            
            
            '2017': [
                'Summer19UL17_V5_MC_L1FastJet_AK4PFchs',
                'Summer19UL17_V5_MC_L2L3Residual_AK4PFchs',
                'Summer19UL17_V5_MC_L2Relative_AK4PFchs',
                'Summer19UL17_V5_MC_L2Residual_AK4PFchs',
                'Summer19UL17_V5_MC_L3Absolute_AK4PFchs',
            ],
            
            '2018': [
                'Summer19UL18_V5_MC_L1FastJet_AK4PFchs',
                'Summer19UL18_V5_MC_L2Relative_AK4PFchs',
                'Summer19UL18_V5_MC_L2L3Residual_AK4PFchs',
                'Summer19UL18_V5_MC_L3Absolute_AK4PFchs',
            ]
        }
        # not updated JUNC
        self._junc = {
            
            '2016': 
            {
                'no_apv': ['Summer19UL16_V7_MC_Uncertainty_AK4PFchs',
                       ],
                'apv': ['Summer19UL16APV_V7_MC_Uncertainty_AK4PFchs',
                    ]},
            
            
            
            '2017': [
                'Summer19UL17_V5_MC_Uncertainty_AK4PFchs',
            ],
            
            '2018': [
                'Summer19UL18_V5_MC_Uncertainty_AK4PFchs',
            ],
        }
        
        self._jr = {
            
            '2016': 
            {'apv': ['Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs',
                 ],
             'no_apv': ['Summer20UL16_JRV3_MC_PtResolution_AK4PFchs',
                    ]},
            
            
            '2017': [
                'Summer19UL17_JRV2_MC_PtResolution_AK4PFchs',
            ],
            
            '2018': [
                'Summer19UL18_JRV2_MC_PtResolution_AK4PFchs',
            ]
        }
        
        self._jersf = {
            
            '2016': 
            {'no_apv':['Summer20UL16_JRV3_MC_SF_AK4PFchs',
                   ],
             'apv':['Summer20UL16APV_JRV3_MC_SF_AK4PFchs',
                ]},
            
            
            
            '2017': [
                'Summer19UL17_JRV2_MC_SF_AK4PFchs',
            ],
            
            '2018': [
                'Summer19UL18_JRV2_MC_SF_AK4PFchs',
            ]
        }
        
        
        self._golden_json_path = {
            '2016': 'data/GoldenJason/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
            '2017': 'data/GoldenJason/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
            '2018': 'data/GoldenJason/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
        }

        self._corrections = corrections
        self._ids = ids
        self._common = common

        self._accumulator = processor.dict_accumulator({
            'sumw': hist.Hist(
                'sumw',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('sumw', 'Weight value', [0.])),
            
            'ele_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_pT', 'Tight electron $p_{T}$ [GeV]', 10, 0, 200)),

            'mu_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_pT', 'Tight Muon $p_{T}$ [GeV]', 10, 0, 200)),
            
            'j1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('j1pt','AK4 Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'j1eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('j1eta','AK4 Jet Eta',35,-3.5,3.5)),
            
            'j1phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('j1phi','AK4 Jet Phi',35,-3.5,3.5)),
            
            'ndflvL': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndflvL', 'AK4 Number of deepFlavor Loose Jets', 6, -0.5, 5.5)),
            'ndflvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndflvM', 'AK4 Number of deepFlavor Medium Jets', 6, -0.5, 5.5)),
            'njets': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('njets', 'AK4 Number of Jets', 10, -0.5, 9.5)),

            'bj1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj1pt','AK4 1st b-Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'bj1eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj1eta','AK4 1st b-Jet Eta',35,-3.5,3.5)),
            
            'bj1phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('bj1phi','AK4 1st b-Jet Phi',35,-3.5,3.5)),
            
            'bj2pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj2pt','AK4 2nd b-Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'bj2eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj2eta','AK4 2nd b-Jet Eta',35,-3.5,3.5)),
            
            'bj2phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('bj2phi','AK4 2nd b-Jet Phi',35,-3.5,3.5)),
            
            'bj3pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj3pt','AK4 3rd b-Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'bj3eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj3eta','AK4 3rd b-Jet Eta',35,-3.5,3.5)),
            
            'bj3phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('bj3phi','AK4 3rd b-Jet Phi',35,-3.5,3.5)),
            
            'bj4pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj4pt','AK4 4th b-Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'bj4eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('bj4eta','AK4 4th b-Jet Eta',35,-3.5,3.5)),
            
            'bj4phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('bj4phi','AK4 4th b-Jet Phi',35,-3.5,3.5)),
            
            'nbjets': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('nbjets', 'AK4 Number of deepCSV Medium BJets', 10, -0.5, 9.5)),

            'ndcsvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndcsvM', 'AK4 Number of deepCSV Medium Jets', 6, -0.5, 5.5)),
            
            'dr_e_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dr_e_lj', '$\Delta r (Leading e, Leading Jet)$', 30, 0, 5.0)),
            'dr_mu_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dr_mu_lj', '$\Delta r (Leading \mu, Leading Jet)$', 30, 0, 5.0)),
            'ele_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_eta', 'Leading Electron Eta', 48, -2.4, 2.4)),
            'mu_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_eta', 'Leading Muon Eta', 48, -2.4, 2.4)),
            'ele_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_phi', 'Leading Electron Phi', 64, -3.2, 3.2)),
            'mu_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_phi', 'Leading Muon Phi', 64, -3.2, 3.2)),

            'dibjetpt' : hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dibjetpt', 'Di-BJet pT', [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])),

            'dibjetmass' : hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dibjetmass', 'Di-BJet Mass',[0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0, 1500, 2000])),

            'dibjeteta'  : hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dibjeteta', 'Di-BJet Mass Eta', 64, -3.2, 3.2)),


            'dibjetphi'  : hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dibjetphi', 'Di-BJet Mass Phi', 64, -3.2, 3.2)),

            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('cut', 'Cut index', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])),
 
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata['dataset']
            
        selected_regions = []
        for region, samples in self._samples.items():
            for sample in samples:
                if sample not in dataset:
                    continue
                selected_regions.append(region)

        isData = 'genWeight' not in events.fields
        selection = processor.PackedSelection()
        hout = self.accumulator.identity()

        ###        # Getting corrections, ids from .coffea files        ###

        if ("preVFP" in dataset) and (self._year == '2016'):
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_preVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_preVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_preVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_preVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_preVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_preVFP_below20'][self._year]
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_preVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_preVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_preVFP'][self._year]
            get_mu_loose_err_sf = self._corrections['get_mu_loose_id_err_preVFP'][self._year]            
            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_preVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_preVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_preVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_preVFP'][self._year]            

            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_preVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_weight_preVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_preVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_preVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_preVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_preVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_preVFP']['deepflav'][self._year]
            
        elif ("postVFP" in dataset) and (self._year == '2016'):
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_postVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_postVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_postVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_postVFP_below20'][self._year]
            
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_postVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_postVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_postVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_postVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_postVFP'][self._year]
            
            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]

        else:
            #  print("hi")
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_postVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_postVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_postVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_postVFP_below20'][self._year]
            
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_postVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_postVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_postVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_postVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_postVFP'][self._year]
            
            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]

        get_msd_weight = self._corrections['get_msd_weight']
        get_ttbar_weight = self._corrections['get_ttbar_weight']
        get_nnlo_nlo_weight = self._corrections['get_nnlo_nlo_weight'][self._year]
        get_nlo_qcd_weight = self._corrections['get_nlo_qcd_weight'][self._year]
        get_nlo_ewk_weight = self._corrections['get_nlo_ewk_weight'][self._year]
        get_pu_weight = self._corrections['get_pu_weight'][self._year]
        get_met_trig_weight = self._corrections['get_met_trig_weight'][self._year]
        get_ele_trig_weight = self._corrections['get_ele_trig_weight'][self._year]
        get_ele_trig_err    = self._corrections['get_ele_trig_err'][self._year]
        get_pho_trig_weight = self._corrections['get_pho_trig_weight'][self._year]
        get_pho_tight_id_sf = self._corrections['get_pho_tight_id_sf'][self._year]
        get_pho_csev_sf = self._corrections['get_pho_csev_sf'][self._year]
        get_ecal_bad_calib = self._corrections['get_ecal_bad_calib']

        isLooseElectron = self._ids['isLooseElectron']
        isTightElectron = self._ids['isTightElectron']
        isLooseMuon = self._ids['isLooseMuon']
        isTightMuon = self._ids['isTightMuon']
        isLooseTau = self._ids['isLooseTau']
        isLoosePhoton = self._ids['isLoosePhoton']
        isTightPhoton = self._ids['isTightPhoton']
        isGoodJet = self._ids['isGoodJet']
        isHEMJet = self._ids['isHEMJet']

        match = self._common['match']
        # to calculate photon trigger efficiency
        sigmoid = self._common['sigmoid']
        deepflavWPs = self._common['btagWPs']['deepflav'][self._year]
        deepcsvWPs = self._common['btagWPs']['deepcsv'][self._year]


        # Initialize physics objects ###
        mu = events.Muon
        n_mu = ak.num(mu)
        mu['isloose'] = isLooseMuon(mu.pt, mu.eta, mu.pfIsoId, mu.looseId, self._year, mu.isPFcand, mu.isGlobal, mu.isTracker)
        mu['istight'] = isTightMuon(mu.pt, mu.eta, mu.pfIsoId, mu.tightId, self._year, mu.isPFcand, mu.isGlobal, mu.isTracker)
        mu_loose = mu[ak.values_astype(mu.isloose, np.bool)]
        mu_tight = mu[ak.values_astype(mu.istight, np.bool)]
        mu_nloose = ak.num(mu_loose)
        mu_ntight = ak.num(mu_tight)
        leading_mu = mu_tight[:,:1]
        
        e = events.Electron
        n_e = ak.num(e)
        #e['isclean'] = ak.all(e.metric_table(mu_loose) >= 0.3, axis=-1)
        e['isclean'] = ~(ak.any(e.metric_table(mu_loose) < 0.3, axis=2))
        e['isloose'] = isLooseElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e['istight'] = isTightElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e_clean = e[ak.values_astype(e.isclean, np.bool)]
        e_loose = e[ak.values_astype(e.isloose, np.bool)]
        e_tight = e_clean[ak.values_astype(e_clean.istight, np.bool)]
        e_nloose = ak.num(e_loose)
        e_ntight = ak.num(e_tight)
        leading_e = e_tight[:,:1]


        tau = events.Tau
        tau['isclean'] = ~(ak.any(tau.metric_table(mu_loose) < 0.4, axis=2)) & ~(ak.any(tau.metric_table(e_loose) < 0.4, axis=2))
        try:
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayMode,tau.idDeepTau2017v2p1VSjet,self._year)
        except:
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayModeOldDMs,tau.idDeepTau2017v2p1VSjet,self._year)
        else: 
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayModeNewDMs,tau.idDeepTau2017v2p1VSjet,self._year)

        tau_clean = tau[ak.values_astype(tau.isclean, np.bool)]
        tau_loose = tau_clean[ak.values_astype(tau_clean.isloose, np.bool)]
        tau_ntot = ak.num(tau, axis=1)
        tau_nloose = ak.num(tau_loose, axis=1)

        pho = events.Photon
        pho_n = ak.num(pho,axis=1)
        pho['isclean'] = ~(ak.any(pho.metric_table(mu_loose) < 0.4, axis=2)) & ~(ak.any(pho.metric_table(e_loose) < 0.4, axis=2))
        pho['isloose'] = isLoosePhoton(pho.pt, pho.eta, pho['cutBased'], self._year)  ## no electronveto version
        pho['istight'] = isTightPhoton(pho.pt, pho['cutBased'], self._year) & (pho.isScEtaEB) & (pho.electronVeto)  # tight photons are barrel only
        pho_clean = pho[ak.values_astype(pho.isclean, np.bool)]
        pho_loose = pho_clean[ak.values_astype(pho_clean.isloose, np.bool)]
        pho_tight = pho_clean[ak.values_astype(pho_clean.istight, np.bool)]
        pho_nloose = ak.num(pho_loose, axis=1)
        pho_ntight = ak.num(pho_tight, axis=1)
        leading_pho = pho_tight[:,:1]
        
        

        j = events.Jet
        j['isgood'] = isGoodJet(j.pt, j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)
        j['isHEM'] = isHEMJet(j.pt, j.eta, j.phi)
        j['isdcsvL'] = (j.btagDeepB>deepcsvWPs['loose'])
        j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose'])
        j['isdflvM'] = (j.btagDeepFlavB > deepflavWPs['medium']) # Link for Twiki
        j['isdcsvM'] = (j.btagDeepB > deepcsvWPs['medium']) #

        j["T"] = ak.zip({"pt": j.pt, "phi": j.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        j['p4'] = ak.zip({
        "pt": j.pt,
        "eta": j.eta,
        "phi": j.phi,
        "mass": j.mass},
        with_name="PtEtaPhiMLorentzVector",
        )
        
        jetMuMask = ak.all(j.metric_table(mu_loose) > 0.4, axis=-1)
        jetEleMask = ak.all(j.metric_table(e_loose) > 0.4, axis=-1)
        jetPhoMask = ak.all(j.metric_table(pho_loose) > 0.4, axis=-1)

        j_isclean_mask = (jetMuMask & jetEleMask & jetPhoMask)
        j_isgood_mask = isGoodJet(j.pt, j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)
        j_good_clean = j[j_isclean_mask & j_isgood_mask]
        j_ngood_clean = ak.num(j_good_clean)
        leading_j = j_good_clean[:,:1] # new way to define leading jet
        #j_good_clean_dflvB = j_good_clean.isdflvM
        print('good clean jet len: ', len(j_good_clean), 'leading jet len: ', len(leading_j), leading_j.pt)
        
        j_good_clean_dflvB = j_good_clean[j_good_clean.isdflvM]
        
        j_ndflvM=ak.num(j_good_clean_dflvB)

        j_HEM = j[ak.values_astype(j.isHEM, np.bool)]       
        j_nHEM = ak.num(j_HEM, axis=1)
        atleast_one_jet_with_pt_grt_50 = ((ak.num(j_good_clean)>=1) & ak.any(j_good_clean.pt>=50, axis=-1))
        print('good clean B jet', j_good_clean_dflvB.pt, len(j_good_clean_dflvB.pt))
        onebjets = j_good_clean_dflvB[:,:1]

        print('first b jet ', onebjets.pt, len(onebjets.pt))
        twobjets = j_good_clean_dflvB[:,1:2]
        print('second b jet', twobjets.pt, len(twobjets.pt))
        threebjets = j_good_clean_dflvB[:,2:3]
        fourbjets = j_good_clean_dflvB[:,3:4]
        dibj = ak.cartesian({"onebj":onebjets,"twobj":twobjets})
        print('dibj', dibj.fields)
        dibjet = dibj.onebj + dibj.twobj
        print('dibjet', dibjet, dibjet.pt)
        
        # *****btag
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X#Supported_Algorithms_and_Operati
        # medium     0.4184
        # btagWP_medium = 0.4184
        # Jet_btag_medium = j_clean[j_clean['btagDeepB'] > btagWP_medium]
        ###
        # Calculating derivatives
        ###

        # *******calculate deltaR( leading ak4jet, e/mu) < 3.4 *****
        LJ_Ele = ak.cartesian({"leading_j":leading_j, "e_loose": e_loose})
        DeltaR_LJ_Ele = abs(LJ_Ele.leading_j.delta_r(LJ_Ele.e_loose))
        
        DeltaR_LJ_Ele_mask = ak.any(DeltaR_LJ_Ele < 3.4, axis=-1)

        LJ_Mu = ak.cartesian({"leading_j":leading_j, "mu_loose": mu_loose})
        DeltaR_LJ_Mu = abs(LJ_Mu.leading_j.delta_r(LJ_Mu.mu_loose))
        
        DeltaR_LJ_Mu_mask = ak.any(DeltaR_LJ_Mu < 3.4, axis=-1)

        if isData:
            lumi_mask = np.array(LumiMask(self._golden_json_path[self._year])(events.run, events.luminosityBlock), dtype=bool)
            
        else: lumi_mask = np.ones(len(events))

        ###
        # Calculating weights
        ###
        if not isData:

            ###
            # JEC/JER
            ###

            gen = events.GenPart

            gen['isb'] = (abs(gen.pdgId) == 5) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])

            gen['isc'] = (abs(gen.pdgId) == 4) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])

            gen['isTop'] = (abs(gen.pdgId) == 6) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            ttjet_weights = np.ones(event_size)
            if('TTJets' in dataset):
                ttjet_weights = np.sqrt(get_ttbar_weight(ak.sum(genTops.pt, axis=-1)) * get_ttbar_weight(ak.sum(genTops.pt, axis=-1)))
                #ttjet_weights = np.sqrt(get_ttbar_weight(genTops.pt.sum()) * get_ttbar_weight(genTops.pt.sum()))

            gen['isW'] = (abs(gen.pdgId) == 24) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isZ'] = (abs(gen.pdgId) == 23) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isA'] = (abs(gen.pdgId) == 22) & gen.hasFlags(['isPrompt', 'fromHardProcess', 'isLastCopy']) & (gen.status == 1)

            ###
            # Calculate PU weight and systematic variations
            ###

            pu = get_pu_weight(events.Pileup.nTrueInt)

            ###
            # Trigger efficiency weight
            ###


            trig = {
                'sr': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
            }
            trig_err = {
                'sr': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
            }

            ###
            # Calculating electron and muon ID weights
            ###
            ids = {
                'sr': get_ele_tight_id_sf(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
            }
            ids_err = {
                'sr': get_ele_tight_id_err(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
            }

            ###
            # Reconstruction weights for electrons
            ###

            # 2017 has separate weights for low/high pT (threshold at 20 GeV)
            def ele_reco_sf(pt, eta): 
                return get_ele_reco_sf(eta, pt)*ak.values_astype((pt > 20), np.int) + get_ele_reco_lowet_sf(eta, pt)*ak.values_astype((~(pt > 20)), np.int)
            
            def ele_reco_err(pt, eta):
                return get_ele_reco_err(eta, pt)*ak.values_astype((pt > 20), np.int) + get_ele_reco_lowet_err(eta, pt)*ak.values_astype((~(pt > 20)), np.int)

            #look at this RISHABH
            if self._year == '2017' or self._year == '2018' or self._year == '2016':
                sf = ele_reco_sf
            else:
                sf = get_ele_reco_sf

            reco = {
                'sr': np.ones(event_size),
            }
            reco_err = {
                'sr': np.ones(event_size),
            }
            ###
            # Isolation weights for muons
            ###

            isolation = {
                'sr': np.ones(event_size),

            }
            isolation_err = {
                'sr': np.ones(event_size),
            }
            ###
            # AK4 b-tagging weights
            ###
            '''
            if in a region you are asking for 0 btags, you have to apply the 0-btag weight
            if in a region you are asking for at least 1 btag, you need to apply the -1-btag weight
            it’s “-1” because we want to reserve “1" to name the weight that should be applied when you ask for exactly 1 b-tag
            that is different from the weight you apply when you ask for at least 1 b-tag
            '''
            #             if 'preVFP' in dataset:
            #                 VFP_status = 'preVFP'
            #             elif 'postVFP' in dataset:
            #                 VFP_status = 'postVFP'
            #             else:
            # #                 VFP_status = False

            btag = {}
            btagUp = {}
            btagDown = {}
            btag['sr'],   btagUp['sr'],   btagDown['sr'] = get_deepflav_weight['medium'](j_good_clean,j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '+1', )



        ###
        # Selections
        ###

        met_filters = np.ones(event_size, dtype=np.bool)

        # this filter is recommended for data only
        if isData:
            met_filters = met_filters & events.Flag['eeBadScFilter']
        for flag in AnalysisProcessor.met_filter_flags[self._year]:
            met_filters = met_filters & events.Flag[flag]
        selection.add('met_filters', ak.to_numpy(met_filters, np.bool))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._jet_triggers[self._year]:
            print('jet trigger1: ', path)
            if path not in events.HLT.fields:
                continue
            print('jet trigger2: ', path)
            triggers = triggers | events.HLT[path]
        selection.add('jet_triggers', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._jet_triggers_sr[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('jet_triggers_sr', ak.to_numpy(triggers))

        noHEMj = np.ones(event_size, dtype=np.bool)
        #if self._year == '2018':
            #noHEMj = (j_nHEM == 0)
        #noHEMmet = np.ones(event_size, dtype=np.bool)
        #if self._year == '2018':
            #noHEMmet = (met.pt > 470) | (met.phi > -0.62) | (met.phi < -1.62)

        selection.add('DeltaR_LJ_mask',ak.to_numpy(DeltaR_LJ_Ele_mask | DeltaR_LJ_Mu_mask))
        selection.add('isoneM', ak.to_numpy((e_nloose == 0) & (mu_ntight == 1) & ( mu_nloose == 1)))
        selection.add('isoneE', ak.to_numpy((e_ntight == 1) & (e_nloose == 1) & (mu_nloose == 0)))

        selection.add('exactly_1_medium_btag', ak.to_numpy(j_ndflvM == 1))
        selection.add('atleast_2_medium_btag', ak.to_numpy(j_ndflvM >= 2))
        selection.add('zero_medium_btags', ak.to_numpy(j_ndflvM == 0))

        selection.add('noHEMj', ak.to_numpy(noHEMj))
        #        selection.add('noHEMmet', ak.to_numpy(noHEMmet))
        selection.add('DeltaR_LJ_Ele_mask', ak.to_numpy((DeltaR_LJ_Ele_mask)>0))

        selection.add('one_muon', ak.to_numpy(ak.num(mu_tight, axis=1) == 1))
        selection.add('zero_loose_electron', ak.to_numpy(ak.num(e_loose, axis=1) == 0))
        selection.add('DeltaR_LJ_Mu_mask', ak.to_numpy((DeltaR_LJ_Mu_mask)>0))
        selection.add('atleast_4_medium_btag', ak.to_numpy(j_ndflvM >= 4))

        selection.add('leading_j>70',ak.to_numpy(ak.sum(leading_j.pt, axis=1) >70)) # 
        selection.add('atleast_one_jet_with_pt_grt_50',ak.to_numpy(atleast_one_jet_with_pt_grt_50))

        
        
        #print(selection.all())
        regions = {
            'sr':['met_filters', 'jet_triggers']
        }
        isFilled = False
        for region, cuts in regions.items():
            if region not in selected_regions: continue
            print('Considering region:', region)

            variables = {

                'mu_pT':               mu_tight.pt,
                'mu_phi' : leading_mu.phi,
                'mu_eta' : leading_mu.eta,
                'ele_pT':              e_tight.pt,
                'ele_phi'     : leading_e.phi,
                'ele_eta'     : leading_e.eta,
                'j1pt':                j_good_clean.pt,
                'j1eta':               j_good_clean.eta,
                'j1phi':               j_good_clean.phi,
                'njets':               j_ngood_clean,
                'e1pt'      : leading_e.pt,
                'nbjets'      : j_ndflvM,
                'dr_e_lj': DeltaR_LJ_Ele,
                'dr_mu_lj': DeltaR_LJ_Mu,
            }

            bj_variables = {
                'bj1pt':                onebjets.pt,
                'bj1eta':               onebjets.eta,
                'bj1phi':               onebjets.phi,
                'bj2pt':                twobjets.pt,
                'bj2eta':               twobjets.eta,
                'bj2phi':               twobjets.phi,
                'bj3pt':                threebjets.pt,
                'bj3eta':               threebjets.eta,
                'bj3phi':               threebjets.phi,
                'bj4pt':                fourbjets.pt,
                'bj4eta':               fourbjets.eta,
                'bj4phi':               fourbjets.phi,
                'dibjetpt' : dibjet.pt,
                'dibjetmass' : dibjet.mass,
                'dibjeteta'   : dibjet.eta,
                'dibjetphi'  : dibjet.phi,
            }

            def fill(dataset, weight, cut):
                print('cut in fill func', cut,len(cut))
                for k, v in variables.items():
                    print('k: ', k, 'v:', v, len(v))
                    #print('k v[cut]:', v[cut])
                    #print('k flatten:', ak.flatten(v[cut], axis=None))
                for kk, vv in bj_variables.items():
                    print('kk: ', kk, 'vv: ', vv, len(vv))
                    #print('kk v[cut]:', vv[vcut])
                    #print('k flatten:', ak.flatten(vv[cut], axis=None))

                flat_variables = {k: ak.flatten(v[cut], axis=None) for k, v in variables.items()}
                #print('flat variables', flat_variables)
                bjet_variables = {kk: ak.flatten(vv[cut], axis=None) for kk, vv in bj_variables.items()}
                #print('bjet variables', bjet_variables)
                flat_weight = {k: ak.flatten(~np.isnan(v[cut])*weight[cut], axis=None) for k, v in variables.items()}
                #print(' variable items: ', variables.items())
                #print('bj variable items: ', bj_variables.items())
                
                #bjet_weight = {kk: ak.flatten(~np.isnan(vv[cut])*weight[cut], axis=None) for kk, vv in bj_variables.items()}
                for histname, h in hout.items():
                    print('fill hist - ', histname, '-->', h)
                    if not isinstance(h, hist.Hist):
                        continue
                    if histname in bj_variables:
                        flat_variable = {histname: ak.flatten(bj_variables[histname])}
                        print('flat variable: ', flat_variable)
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable)
                               #weight=bjet_weight[histname])
                    if histname not in variables:
                        continue
                    elif histname == 'sumw':
                        continue
                    else:
                        flat_variable = {histname: flat_variables[histname]}
                        print('flat variable: ', flat_variable)
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable,
                               weight=flat_weight[histname])
                        
            if isData:
                if not isFilled:
                    hout['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                    isFilled = True
                cut = selection.all(*regions[region])
                fill(dataset, np.ones(event_size), cut)
            else:
                weights = Weights(len(events))
                if 'L1PreFiringWeight' in events.fields:
                    weights.add('prefiring', events.L1PreFiringWeight.Nom)
                weights.add('genw', events.genWeight)
#                weights.add('nlo_qcd', nlo_qcd)
#                weights.add('nlo_ewk', nlo_ewk)
                #weights.add('ttjet_weights', ttjet_weights)
                weights.add('pileup', pu)
                if 'WJets' in dataset or 'DY' in dataset or 'ZJets' in dataset or 'GJets' in dataset:
                    if not isFilled:
                        hout['sumw'].fill(dataset='HF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        hout['sumw'].fill(dataset='LF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        isFilled = True
                    whf = ak.values_astype(((ak.num(gen[gen.isb],axis=1) > 0) | (ak.num(gen[gen.isc], axis=1) > 0)), np.int)
                    wlf = ak.values_astype(~(ak.values_astype(whf,np.bool)), np.int)
                    cut = ak.to_numpy(selection.all(*regions[region]))
#                     import sys
#                     print(weights._modifiers.keys())
#                     sys.exit(0)
                    if 'WJets' in dataset:
                        systematics = [None,
#                                    'btagUp',
#                                    'btagDown',
                                       trig_name+'Up', trig_name+'Down',
                                       ids_name+'Up', ids_name+'Down',
                                       reco_name+'Up', reco_name+'Down',
                                       isolation_name+'Up', isolation_name+'Down',
                                      ]
                    else:
                        systematics = [None,
#                                        'btagUp',
#                                        'btagDown',
                                       'qcd1Up',
                                       'qcd1Down',
                                       'qcd2Up',
                                       'qcd2Down',
                                       'qcd3Up',
                                       'qcd3Down',
                                       'muFUp',
                                       'muFDown',
                                       'muRUp',
                                       'muRDown',
                                       'ew1Up',
                                       'ew1Down',
                                       'ew2GUp',
                                       'ew2GDown',
                                       'ew2WUp',
                                       'ew2WDown',
                                       'ew2ZUp',
                                       'ew2ZDown',
                                       'ew3GUp',
                                       'ew3GDown',
                                       'ew3WUp',
                                       'ew3WDown',
                                       'ew3ZUp',
                                       'ew3ZDown',
                                       'mixUp',
                                       'mixDown',
                                       trig_name+'Up', trig_name+'Down',
                                       ids_name+'Up', ids_name+'Down',
                                       reco_name+'Up', reco_name+'Down',
                                       isolation_name+'Up', isolation_name+'Down',
                                      ]
                    for systematic in systematics:
                        sname = 'nominal' if systematic is None else systematic
                    ## Cutflow loop
                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*whf)
                    hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*wlf)
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*whf)
                        hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*wlf)
                    fill('HF--'+dataset, weights.weight()*whf, cut)
                    fill('LF--'+dataset, weights.weight()*wlf, cut)

                else:
                    if not isFilled:
                        hout['sumw'].fill(dataset=dataset, sumw=1, weight=ak.sum(events.genWeight, axis=-1))
                        isFilled = True
                    cut = selection.all(*regions[region])
                    for systematic in [None]:
#                    for systematic in [None,
#                                    'btagUp',
#                                    'btagDown',
#                                       trig_name+'Up', trig_name+'Down',
#                                       ids_name+'Up', ids_name+'Down',
#                                       reco_name+'Up', reco_name+'Down',
#                                       isolation_name+'Up', isolation_name+'Down',
#                                      ]:
                        sname = 'nominal' if systematic is None else systematic
                        ## Cutflow loop
                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight())
                    allcuts = set()
                    print('before fill dataset', dataset)
                    print('before fill weight', weights.weight(), len(weights.weight()))
                    print('before fill cut', cut, len(cut))
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        print('cut', i, 'sel:', icut)
                        hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight()*jcut)
                    fill(dataset, weights.weight(), cut)
#         time.sleep(0.5)
        return hout

    def postprocess(self, accumulator):
        scale = {}
        for d in accumulator['sumw'].identifiers('dataset'):
            print('Scaling:', d.name)
            dataset = d.name
            if '--' in dataset:
                dataset = dataset.split('--')[1]
            print('Cross section:', self._xsec[dataset])
            if self._xsec[dataset] != -1:
#                 scale[d.name] = 1
                scale[d.name] = self._lumi*self._xsec[dataset]
            else:
                scale[d.name] = 1

        for histname, h in accumulator.items():
            if histname == 'sumw':
                continue
            if isinstance(h, hist.Hist):
                h.scale(scale, axis='dataset')

        return accumulator


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    (options, args) = parser.parse_args()

    #with open('metadata/UL_'+options.year+'.json') as fin:
    with open('metadata/BFF_T2_KISTI_2017_v1.json') as fin:
    #with open('metadata/Signal.json') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k, v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    ids = load('data/ids.coffea')
    common = load('data/common.coffea')

    processor_instance = AnalysisProcessor(year=options.year,
                                           xsec=xsec,
                                           corrections=corrections,
                                           ids=ids,
                                           common=common)

    save(processor_instance, 'data/UL_BFF_Final_v1_'+options.year+'.processor')
    print("processor name: UL_BFF_Final_v1_{}".format(options.year))
