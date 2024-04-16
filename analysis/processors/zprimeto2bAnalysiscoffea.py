#!/usr/bin/env python                                                                                                                                                                                                           
import logging
import numpy as np
import awkward as ak
import json
import copy
from collections import defaultdict
from coffea import processor
import hist
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from coffea.util import load, save
from optparse import OptionParser
from coffea.nanoevents.methods import vector
import gzip

def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

class AnalysisProcessor(processor.ProcessorABC):

    lumis = {  # Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/TWikiLUM
        '2016postVFP': 16.81,
        '2016preVFP': 19.5,
        '2017': 41.48,
        '2018': 59.83

        #'2016': 16.81, #postVFP
        #'2017': 41.48,
        #'2017':0.0791,
        #'2018': 59.83
    }

    lumiMasks = {
        '2016postVFP': LumiMask("data/jsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        '2016preVFP': LumiMask("data/jsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        '2017': LumiMask("data/jsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
        '2018"': LumiMask("data/jsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
    }


    met_filter = {
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2 
        '2016postVFP': [
            'goodVertices',
            'globalSuperTightHalo2016Filter',
            'HBHENoiseFilter',
            'HBHENoiseIsoFilter',
            'EcalDeadCellTriggerPrimitiveFilter',
            'BadPFMuonFilter',
            'BadPFMuonDzFilter',
            'eeBadScFilter'
        ],
        
        '2016preVFP': [
            'goodVertices',
            'globalSuperTightHalo2016Filter',
            'HBHENoiseFilter',
            'HBHENoiseIsoFilter',
            'EcalDeadCellTriggerPrimitiveFilter',
            'BadPFMuonFilter',
            'BadPFMuonDzFilter',
            'eeBadScFilter'
        ],
        
        '2017': [
            'goodVertices',
            'globalSuperTightHalo2016Filter',
            'HBHENoiseFilter',
            'HBHENoiseIsoFilter',
            'EcalDeadCellTriggerPrimitiveFilter',
            'BadPFMuonFilter',
            'BadPFMuonDzFilter',
            'eeBadScFilter',
            'ecalBadCalibFilter'
        ],
        
        '2018': [
            'goodVertices',
            'globalSuperTightHalo2016Filter',
            'HBHENoiseFilter',
            'HBHENoiseIsoFilter',
            'EcalDeadCellTriggerPrimitiveFilter',
            'BadPFMuonFilter',
            'BadPFMuonDzFilter',
            'eeBadScFilter',
            'ecalBadCalibFilter'
        ]
    }

    def __init__(self, year, xsec, corrections, ids, common):
        self._year = year
        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])
        self._xsec = xsec
         self._systematics = True
        self._skipJER = False

        self._samples = {
            'cr':('TT','QCD', 'Zprime', 'BT'),
            'sr':('TT','QCD', 'Zprime'),
        }

        self._jet_triggers_sr = {
            '2016': [
                'DoubleJet90_Double30_TripleBTagCSV_p087',
                'HLT_QuadJet45_TripleBTagCSV_p087',
            ],
            '2017':[
                'PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0',
            ],
            '2018':[
                'DoublePFJets40_CaloBTagCSV',
                'PFHT180'
            ]
        }
        self._jet_triggers_cr = {
            '2016': [
                'DoubleJet90_Double30_TripleBTagCSV_p087',
                'HLT_QuadJet45_TripleBTagCSV_p087'
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
    
        self._corrections = corrections
        self._ids = ids
        self._common = common
        ptbins=[30.0, 60.0,  90.0,  120.0,  150.0,  180.0,  210.0,  250.0,  280.0,  310.0,  340.0,  370.0,  400.0,  430.0,  470.0,  510.0,  550.0,  590.0,  640.0,  690.0,  740.0,  790.0,  840.0,  900.0,  960.0,  1020.0,  1090.0,  1160.0,  1250.0]

        self.make_output = lambda: {
            'sumw': 0.,
            'ele_pT': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='ele_pT', label='Leading Electron Pt'),
                storage=hist.storage.Weight(),
            ),
                
            'mu_pT': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='mu_pT', label='Leading Muon Pt'),
                storage=hist.storage.Weight(),
            ),
              
            'j1pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='j1pt', label='AK4 Leading Jet Pt'),
                storage=hist.storage.Weight(),
            ),
            
            'j1eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='j1eta', label='AK4 Leading Jet Eta'),
                storage=hist.storage.Weight(),
            ),

            'j1phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='j1phi', label='AK4 Leading Jet Phi'),
                storage=hist.storage.Weight(),
            ),
            
            'ndflvL': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name='ndflvL', label='AK4 Number of deepFlavor Loose Jets'),
                storage=hist.storage.Weight(),
            ),
            'ndflvM': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name='ndflvM', label='AK4 Number of deepFlavor Medium Jets'),
                storage=hist.storage.Weight(),
            ),
            'njets': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name='njets', label='AK4 Number of Jets'),
                storage=hist.storage.Weight(),
            ),
            
            'nbjets': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name='bnjets', label='AK4 Number of b-Jets'),
                storage=hist.storage.Weight(),
            ),
            
            'bj1pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='bj1pt', label='AK4 Leading b-Jet Pt'),
                storage=hist.storage.Weight(),
                ),

            'bj1eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj1eta', label='AK4 Leading b-Jet Eta'),
                storage=hist.storage.Weight(),
            ),
            'bj1phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj1phi', label='AK4 Leading b-Jet Phi'),
                storage=hist.storage.Weight(),
            ),
        
            'bj2pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='bj2pt', label='AK4 subleading Jet Pt'),
                storage=hist.storage.Weight(),
            ),
            'bj2eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj2eta', label='AK4 subleading b-Jet Eta'),
                storage=hist.storage.Weight(),
            ),
            'bj2phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj2phi', label='AK4 subleading b-Jet Phi'),
                storage=hist.storage.Weight(),
            ),

            'bj3pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='bj3pt', label='AK4 3rd leading Jet Pt'),
                storage=hist.storage.Weight(),

            ),
            'bj3eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj3eta', label='AK4 3rd leading b-Jet Eta'),
                storage=hist.storage.Weight(),
            ),
            'bj3phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj3phi', label='AK4 3rd leading b-Jet Phi'),
                storage=hist.storage.Weight(),
            ),


            'bj4pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='bj4pt', label='AK4 4th leading Jet Pt'),
                storage=hist.storage.Weight(),

            ),
            'bj4eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj4eta', label='AK4 4th leading b-Jet Eta'),
                storage=hist.storage.Weight(),
            ),
            'bj4phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='bj4phi', label='AK4 4th leading b-Jet Phi'),
                storage=hist.storage.Weight(),
            ),
            'ndcsvM': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name='ndcsvM', label='AK4 Number of deepCSV Medium Jets'),
                storage=hist.storage.Weight(),
            ),
            
            'dr_e_lj': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(30,0,5.0, name='dr_e_lj', label='$\Delta r (Leading e, Leading Jet)$'),
                storage=hist.storage.Weight(),
            ),
            'dr_mu_lj': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(30,0,5.0, name='dr_m_lj', label='$\Delta r (Leading muon, Leading Jet)$'),
                storage=hist.storage.Weight(),
            ),

            'ele_eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='ele_eta', label='Electron Eta'),
                storage=hist.storage.Weight(),

            ),
            'mu_eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='mu_eta', label='MuonEta'),
                storage=hist.storage.Weight(),

            ),

            'ele_phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='ele_phi', label='Electron Phi'),
                storage=hist.storage.Weight(),
            ),
            'mu_phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='mu_phi', label='MuonPhi'),
                storage=hist.storage.Weight(),
            ),

            'dibjetpt' : hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Variable(ptbins, name='dibjetpt', label='Di-BJet pT'),
                storage=hist.storage.Weight(),
            ),
            
            'dibjetmass' : hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(20,0,600, name='dibjetmass', label='Di-BJet Mass'),
                storage=hist.storage.Weight(),
            ),

            'dibjeteta'  : hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='dibjeteta', label='Di-BJet Eta'),
                storage=hist.storage.Weight(),
            ),

            'dibjetphi'  : hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(35,-3.5,3.5, name='ele_phi', label='Di-BJet Phi'),
                storage=hist.storage.Weight(),
            ),

        }


    def process(self, events):
        isData = not hasattr(events, "genWeight")
        if isData:
            # Nominal JEC are already applied in data 
            return self.process_shift(events, None)
        jet_factory              = self._corrections['jet_factory']
        subjet_factory           = self._corrections['subjet_factory']
        met_factory              = self._corrections['met_factory']

        import cachetools
        jec_cache = cachetools.Cache(np.inf)

        nojer = "NOJER" if self._skipJER else ""
        thekey = f"{self._year}mc{nojer}"

        def add_jec_variables(jets, event_rho):
            jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
            jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
            return jets

        jets = jet_factory[thekey].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache)
        #fatjets = fatjet_factory[thekey].build(add_jec_variables(events.AK15PFPuppiJet, events.fixedGridRhoFastjetAll), jec_cache) 
        subjets = subjet_factory[thekey].build(add_jec_variables(events.AK15PFPuppiSubJet, events.fixedGridRhoFastjetAll), jec_cache)
        met = met_factory.build(events.MET, jets, {})
        
        def process_shift(self, events, shift_name):
            dataset = events.metadata['dataset']

            selected_regions = []
            for region, samples in self._samples.items():
                for sample in samples:
                    if sample not in dataset: continue
                    selected_regions.append(region)
                    
            isData = not hasattr(events, "genWeight")
            selection = PackedSelection(dtype="uint64")
            weights = Weights(len(events), storeIndividual=True)
            output = self.make_output()
            if shift_name is None and not isData:
            output['sumw'] = ak.sum(events.genWeight)

            ###
            #Getting corrections, ids from .coffea files 
            ###
            get_met_trig_weight      = self._corrections['get_met_trig_weight']
            get_ele_loose_id_sf      = self._corrections['get_ele_loose_id_sf']
            get_ele_tight_id_sf      = self._corrections['get_ele_tight_id_sf']
            get_ele_trig_weight      = self._corrections['get_ele_trig_weight']
            get_ele_reco_sf_below20  = self._corrections['get_ele_reco_sf_below20']
            get_ele_reco_sf_above20  = self._corrections['get_ele_reco_sf_above20']
            get_mu_loose_id_sf       = self._corrections['get_mu_loose_id_sf']
            get_mu_tight_id_sf       = self._corrections['get_mu_tight_id_sf']
            get_mu_loose_iso_sf      = self._corrections['get_mu_loose_iso_sf']
            get_mu_tight_iso_sf      = self._corrections['get_mu_tight_iso_sf']
            get_mu_rochester_sf      = self._corrections['get_mu_rochester_sf'][self._year]
            get_met_xy_correction    = self._corrections['get_met_xy_correction']
            get_pu_weight            = self._corrections['get_pu_weight']
            get_nlo_ewk_weight       = self._corrections['get_nlo_ewk_weight']
            get_nnlo_nlo_weight      = self._corrections['get_nnlo_nlo_weight'][self._year]
            get_msd_corr             = self._corrections['get_msd_corr']
            get_btag_weight      = self._corrections['get_btag_weight']
            
            isLooseElectron = self._ids['isLooseElectron']
            isTightElectron = self._ids['isTightElectron']
            isLooseMuon     = self._ids['isLooseMuon']
            isTightMuon     = self._ids['isTightMuon']
            isLooseTau      = self._ids['isLooseTau']
            isLoosePhoton   = self._ids['isLoosePhoton']
            isTightPhoton   = self._ids['isTightPhoton']
            isGoodAK4       = self._ids['isGoodAK4']
            isHEMJet        = self._ids['isHEMJet']

            deepflavWPs = self._common['btagWPs']['deepflav'][self._year]
            deepcsvWPs = self._common['btagWPs']['deepcsv'][self._year]

            ###
            #Initialize global quantities (MET ecc.) 
            ###

            npv = events.PV.npvsGood
            run = events.run
            calomet = events.CaloMET
            met = events.MET
            met['pt'] , met['phi'] = get_met_xy_correction(self._year, npv, run, met.pt, met.phi, isData)

            mu = events.Muon
            if isData:
                k = get_mu_rochester_sf.kScaleDT(mu.charge, mu.pt, mu.eta, mu.phi)
            else:
                kspread = get_mu_rochester_sf.kSpreadMC(
                    mu.charge,
                    mu.pt,
                    mu.eta,
                    mu.phi,
                    mu.matched_gen.pt
                )
                mc_rand = ak.unflatten(np.random.rand(ak.count(ak.flatten(mu.pt))), ak.num(mu.pt))
                ksmear = get_mu_rochester_sf.kSmearMC(
                    mu.charge,
                    mu.pt,
                    mu.eta,
                    mu.phi,
                    mu.nTrackerLayers,
                    mc_rand
                )
                hasgen = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))
                k = ak.where(hasgen, kspread, ksmear)
            mu['pt'] = ak.where((mu.pt<200),k*mu.pt, mu.pt)
            mu['isloose'] = isLooseMuon(mu,self._year)
            mu['id_sf'] = ak.where(
                mu.isloose,
                get_mu_loose_id_sf(self._year, abs(mu.eta), mu.pt),
                ak.ones_like(mu.pt)
            )
            mu['iso_sf'] = ak.where(
                mu.isloose,
                get_mu_loose_iso_sf(self._year, abs(mu.eta), mu.pt),
                ak.ones_like(mu.pt)
            )
            mu['istight'] = isTightMuon(mu,self._year)
            mu['id_sf'] = ak.where(
                mu.istight,
                get_mu_tight_id_sf(self._year, abs(mu.eta), mu.pt),
                mu.id_sf
            )
            mu['iso_sf'] = ak.where(
                mu.istight,
                get_mu_tight_iso_sf(self._year, abs(mu.eta), mu.pt),
                mu.iso_sf
            )
            mu['T'] = ak.zip(
                {
                    "r": mu.pt,
                    "phi": mu.phi,
                },
                with_name="PolarTwoVector",
                behavior=vector.behavior,
            )
            mu_loose=mu[mu.isloose]
            mu_tight=mu[mu.istight]
            mu_ntot = ak.num(mu, axis=1)
            mu_nloose = ak.num(mu_loose, axis=1)
            mu_ntight = ak.num(mu_tight, axis=1)
            leading_mu = ak.firsts(mu_tight)
            
            e = events.Electron
            e['isclean'] = ak.all(e.metric_table(mu_loose) > 0.3, axis=2)
            e['reco_sf'] = ak.where(
                (e.pt<20),
                get_ele_reco_sf_below20(self._year, e.eta+e.deltaEtaSC, e.pt),
                get_ele_reco_sf_above20(self._year, e.eta+e.deltaEtaSC, e.pt)
            )
            e['isloose'] = isLooseElectron(e,self._year)
            e['id_sf'] = ak.where(
                e.isloose,
                get_ele_loose_id_sf(self._year, e.eta+e.deltaEtaSC, e.pt),
                ak.ones_like(e.pt)
            )
            e['istight'] = isTightElectron(e,self._year)
            e['id_sf'] = ak.where(
                e.istight,
                get_ele_tight_id_sf(self._year, e.eta+e.deltaEtaSC, e.pt),
                e.id_sf
            )
            e['T'] = ak.zip(
                {
                    "r": e.pt,
                    "phi": e.phi,
                },
                with_name="PolarTwoVector",
                behavior=vector.behavior,
            )
            e_clean = e[e.isclean]
            e_loose = e_clean[e_clean.isloose]
            e_tight = e_clean[e_clean.istight]
            e_ntot = ak.num(e, axis=1)
            e_nloose = ak.num(e_loose, axis=1)
            e_ntight = ak.num(e_tight, axis=1)
            leading_e = ak.firsts(e_tight)
            
            tau = events.Tau
            tau['isclean']=(
                ak.all(tau.metric_table(mu_loose) > 0.4, axis=2)
                & ak.all(tau.metric_table(e_loose) > 0.4, axis=2)
            )
            
            tau['isloose']=isLooseTau(tau, self._year)
            tau_clean=tau[tau.isclean]
            tau_loose=tau_clean[tau_clean.isloose]
            tau_ntot=ak.num(tau, axis=1)
            tau_nloose=ak.num(tau_loose, axis=1)
            
            pho = events.Photon
            pho['isclean']=(
                ak.all(pho.metric_table(mu_loose) > 0.5, axis=2)
                & ak.all(pho.metric_table(e_loose) > 0.5, axis=2)
                & ak.all(pho.metric_table(tau_loose) > 0.5, axis=2)
            )
            j = events.Jet
            j['isgood'] = isGoodAK4(j, self._year)
            j['isHEM'] = isHEMJet(j)
            j['isclean'] = (
                ak.all(j.metric_table(mu_loose) > 0.4, axis=2)
                & ak.all(j.metric_table(e_loose) > 0.4, axis=2)
                & ak.all(j.metric_table(tau_loose) > 0.4, axis=2)
                & ak.all(j.metric_table(pho_loose) > 0.4, axis=2)
            )
            #j['isiso'] = ak.all(j.metric_table(leading_fj) > 1.5, axis=2)
            j['isdcsvL'] = (j.btagDeepB>deepcsvWPs['loose'])
            j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose'])
            j['T'] = ak.zip(
                {
                    "r": j.pt,
                    "phi": j.phi,
                },
                with_name="PolarTwoVector",
                behavior=vector.behavior,
            )
            
            j_good = j[j.isgood]
            j_clean = j_good[j_good.isclean]
            #j_iso = j_clean[j_clean.isiso]
            j_dcsvL = j_iso[j_iso.isdcsvL]
            j_dflvL = j_iso[j_iso.isdflvL]
            j_HEM = j[j.isHEM]
            j_ntot=ak.num(j, axis=1)
            j_ngood=ak.num(j_good, axis=1)
            j_nclean=ak.num(j_clean, axis=1)
            j_niso=ak.num(j_iso, axis=1)
            j_ndcsvL=ak.num(j_dcsvL, axis=1)
            j_ndflvL=ak.num(j_dflvL, axis=1)
            j_nHEM = ak.num(j_HEM, axis=1)
            leading_j = ak.firsts(j_clean)

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

        ###
        #Calculating weights
        ###
        if not isData:
            
            gen = events.GenPart

            gen['isb'] = (abs(gen.pdgId)==5)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isc'] = (abs(gen.pdgId)==4)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isTop'] = (abs(gen.pdgId)==6)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            nlo = np.ones(len(events), dtype='float')
            if('TTJets' in dataset): 
                nlo = np.sqrt(get_ttbar_weight(genTops[:,0].pt) * get_ttbar_weight(genTops[:,1].pt))

        ###
        # Calculate PU weight and systematic variations
        ###
        pu = get_pu_weight(self._year, events.Pileup.nTrueInt)
        ###

        ### 
        # Calculating electron and muon ID weights
        ###
        
        ids ={
            'sr':  np.ones(len(events), dtype='float'),
            'wmcr': leading_mu.id_sf,
            'tmcr': leading_mu.id_sf,
            'wecr': leading_e.id_sf,
            'tecr': leading_e.id_sf,
            'qcdcr': np.ones(len(events), dtype='float'),
            'qcdcr': np.ones(len(events), dtype='float'),
        }
        
        ###
        # Reconstruction weights for electrons
        ###
        
        reco = {
            'sr': np.ones(len(events), dtype='float'),
            'qcdcr': np.ones(len(events), dtype='float'),
        }
        
        ###
        # Isolation weights for muons
        ###
        
        '''
        isolation = {
        'sr': np.ones(len(events), dtype='float'),
        'qcdcr': np.ones(len(events), dtype='float'),
        }
        '''
        
        ###
        # AK4 b-tagging weights
        ###
        
        btagSF, \
            btagSFbc_correlatedUp, \
            btagSFbc_correlatedDown, \
            btagSFbc_uncorrelatedUp, \
            btagSFbc_uncorrelatedDown, \
            btagSFlight_correlatedUp, \
            btagSFlight_correlatedDown, \
            btagSFlight_uncorrelatedUp, \
            btagSFlight_uncorrelatedDown  = get_btag_weight('deepflav',self._year,'loose').btag_weight(
                j_clean.pt,
                j_clean.eta,
                j_clean.hadronFlavour,
                j_clean.isdflvL
            )

        if hasattr(events, "L1PreFiringWeight"): 
            weights.add('prefiring', events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
        weights.add('genw',events.genWeight)
        weights.add('pileup',pu)
        weights.add('trig', trig[region])
        weights.add('ids', ids[region])
        weights.add('reco', reco[region])
        #weights.add('isolation', isolation[region])
        weights.add('btagSF',btagSF)
        weights.add('btagSFbc_correlated',np.ones(len(events), dtype='float'), btagSFbc_correlatedUp/btagSF, btagSFbc_correlatedDown/btagSF)
        weights.add('btagSFbc_uncorrelated',np.ones(len(events), dtype='float'), btagSFbc_uncorrelatedUp/btagSF, btagSFbc_uncorrelatedDown/btagSF)
        weights.add('btagSFlight_correlated',np.ones(len(events), dtype='float'), btagSFlight_correlatedUp/btagSF, btagSFlight_correlatedDown/btagSF)
        weights.add('btagSFlight_uncorrelated',np.ones(len(events), dtype='float'), btagSFlight_uncorrelatedUp/btagSF, btagSFlight_uncorrelatedDown/btagSF)
        
        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._jet_triggers_sr[self._year]:
            print('jet trigger1: ', path)
            if path not in events.HLT.fields:
                continue
            print('jet trigger2: ', path)
            triggers = triggers | ~events.HLT[path]
        selection.add('no_jet_triggers_sr', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._jet_triggers_cr[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | ~events.HLT[path]
        selection.add('no_jet_triggers_cr', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._jet_triggers_sr[self._year]:
            print('jet trigger1: ', path)
            if path not in events.HLT.fields:
                continue
            print('jet trigger2: ', path)
            triggers = triggers | events.HLT[path]
        selection.add('jet_triggers_sr', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._jet_triggers_cr[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('jet_triggers_cr', ak.to_numpy(triggers))


        noHEMj = np.ones(len(events), dtype='bool')
        if self._year=='2018': noHEMj = (j_nHEM==0)

        lumimask = np.ones(len(events), dtype='bool')
        if isData:
            lumimask = AnalysisProcessor.lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumimask', lumimask)

        met_filters =  np.ones(len(events), dtype='bool')
        #if isData: met_filters = met_filters & events.Flag['eeBadScFilter']#this filter is recommended for data only
        for flag in AnalysisProcessor.met_filters[self._year]:
            met_filters = met_filters & events.Flag[flag]
        selection.add('met_filters',met_filters)
        
        if self._year=='2018': noHEMmet = (met.pt>470)|(met.phi>-0.62)|(met.phi<-1.62)

        selection.add('DeltaR_LJ_mask',ak.to_numpy(DeltaR_LJ_Ele_mask | DeltaR_LJ_Mu_mask))
        selection.add('isoneM', ak.to_numpy((e_nloose == 0) & (mu_ntight == 1) & ( mu_nloose == 1)))
        selection.add('isoneE', ak.to_numpy((e_ntight == 1) & (e_nloose == 1) & (mu_nloose == 0)))

        selection.add('exactly_1_medium_btag', ak.to_numpy(j_ndflvM == 1))
        selection.add('atleast_2_medium_btag', ak.to_numpy(j_ndflvM >= 2))
        selection.add('zero_medium_btags', ak.to_numpy(j_ndflvM == 0))
        selection.add('atleast_4_medium_btag', ak.to_numpy(j_ndflvM >= 4))

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
            'cr':['met_filters', 'jet_triggers_cr','no_jet_triggers_sr','atleast_4_medium_btag'],
            'sr':['met_filters', 'jet_triggers_sr','no_jet_triggers_cr']
        }

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
                
        def fill(region, systematic):
            cut = selection.all(*regions[region])
            sname = 'nominal' if systematic is None else systematic
            if systematic in weights.variations:
                weight = weights.weight(modifier=systematic)[cut]
            else:
                weight = weights.weight()[cut]
            output['template'].fill(
                  region=region,
                  systematic=sname,
                  recoil=normalize(u[region].r, cut),
                  fjmass=normalize(leading_fj.msd_corr, cut),
                  TvsQCD=normalize(leading_fj.TvsQCD, cut),
                  weight=weight
            )
            if systematic is None:
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
            for variable in output:
                    if variable not in variables:
                        continue
                    normalized_variable = {variable: normalize(variables[variable],cut)}
                    output[variable].fill(
                        region=region,
                        **normalized_variable,
                        weight=weight,
                    )

                    
        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
            
        for region in regions:
            if region not in selected_regions: continue
            for systematic in systematics:
                if isData and systematic is not None:
                    continue
                fill(region, systematic)


        scale = 1
        if self._xsec[dataset]!= -1: 
            scale = self._lumi*self._xsec[dataset]

        for key in output:
            if key=='sumw': 
                continue
            output[key] *= scale
                
        return output

########

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-n', '--name', help='name', dest='name')
    (options, args) = parser.parse_args()


    with gzip.open('metadata/'+options.metadata+'.json.gz') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k,v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    ids         = load('data/ids.coffea')
    common      = load('data/common.coffea')

    processor_instance=AnalysisProcessor(year=options.year,
                                         xsec=xsec,
                                         corrections=corrections,
                                         ids=ids,
                                         common=common)

    save(processor_instance, 'data/hadmonotop'+options.name+'.processor')
