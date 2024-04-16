import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import coffea
from coffea import util
import hist
import os, sys
import argparse
from tabulate import tabulate
import uncertainties as unc  
import uncertainties.unumpy as unp

## read merged file
parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str, help="input file")
parser.add_argument("--y", type=str, help="year", default="2017", choices=["2016", "2017", "2018"])
args = parser.parse_args()

class Mydrawer:
    def __init__(self, f, y):
        try:
            self.f = util.load(f)
            self.y = y
        except:
            print("Please provide a valid file. Exiting...")
            sys.exit(1)
        self.fname = str(f).split("/")[-1].split(".")[0]
        self.workingdir = os.getcwd()
        self.outdir = self.workingdir + "/plots/" + self.fname
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
    
    def btagSpliter(self):
        hists = self.f
        year = self.y
        keys = hists.keys()
        deepflav = hists["deepflav"]
        deepcsv = hists["deepcsv"]
        if year == "2016":
            btag = deepcsv
        else:
            btag = deepflav
        for mcset in hists['deepflav']:
            try:
                btag += hists['deepflav'][mcset]
            except:
                btag = hists['deepflav'][mcset]
            bpass = btag[{"wp": "loose", "btag": "pass"}].view()
            ball = btag[{"wp": "loose", "btag": sum}].view()
            upass = bpass
            uall = ball
            # Light Flavor Jets
            lfpass = unp.uarray((bpass[0], np.sqrt(upass[0])))
            lfall = unp.uarray((ball[0], np.sqrt(uall[0])))
            lfall[lfall <= 0] = 1.00
            lf = lfpass / lfall
            # Charm Flavor Jets
            cfpass = unp.uarray((bpass[1], np.sqrt(upass[1])))
            cfall = unp.uarray((ball[1], np.sqrt(uall[1])))
            cfall[cfall <= 0] = 1.00
            cf = cfpass / cfall
            # Bottom Flavor Jets
            bfpass = unp.uarray((bpass[2], np.sqrt(upass[2])))
            bfall = unp.uarray((ball[2], np.sqrt(uall[2])))
            bfall[bfall <= 0] = 1.00
            bf = bfpass / bfall
            return lf, cf, bf

    def btagDraw(self):
        year = self.y
        yedges = [30, 50, 70, 100, 140, 200, 300, 600, 1000]
        xedges = [0, 1.4, 2.0, 2.5]
        ybins = np.array(yedges)# + np.diff(yedges)/2
        xbins = np.array(xedges)# + np.diff(xedges)/2
        nameset = ['lf', 'cf', 'bf']
        idx = 0
        for flavor in self.btagSpliter():
            eff=np.array(unp.nominal_values(flavor[1:,:]))
            #eff=np.flip(eff, axis=0)
            xhist= []
            yhist= []
            weight_2d = []
            print(eff)
            for y in range(len(yedges)-1):
                for x in range(len(xedges)-1):
                    print(x,y)
                    xhist.append((xedges[x]+xedges[x+1])/2)
                    yhist.append((yedges[y]+yedges[y+1])/2)
                    weight_2d.append(eff[y][x])
            plt.style.use(hep.style.CMS)
            fig, ax = plt.subplots(1,1,figsize=(10,8))
            hep.cms.label(ax=ax, llabel='Private Work', rlabel=str(year)+' (13 TeV)')
            hist, xbins, ybins, im = plt.hist2d(xhist, yhist, weights=weight_2d, bins=[xbins, ybins], cmap='jet', cmin=0,cmax=1.0)
            for i in range(len(ybins)-1):
                for j in range(len(xbins)-1):
                    ax.text((xbins[j+1] - xbins[j])/2 + xbins[j], (ybins[i+1] - ybins[i])/2 + ybins[i], 
                            np.round(hist.T[i,j],2), 
                            color="w", ha="center", va="top", fontweight="bold")
            # colormap
            plt.colorbar()
            # transpose the array
            ax.set_xlabel(r'Jet $|\eta|$')
            ax.set_ylabel(r'Jet $p_{T}$ [GeV]')
            ax.set_ylim(30,1000)
            ax.set_xlim(0,2.5)
            ax.set_yscale('log')
            plt.savefig(self.outdir+'/btageff_2d_'+nameset[idx]+'.png')
            plt.close()
            idx += 1

if __name__ == "__main__":
    drawer = Mydrawer(args.i, args.y)
    drawer.btagDraw()
    print("done")
