import numpy as np
from pylab import plt


def indexLayers(sectionNameToDepth):
    indexSection = lambda sectionName, depth: [(sectionName, i + 1) for i in range(depth)]
    sectionIndexLists = [indexSection(sectionName, depth) for sectionName, depth in sectionNameToDepth.items()]
    layerIndices = sum(sectionIndexLists, [])
    return layerIndices


class FindKey:
    def __init__(self, allKeys):
        self.allKeys = allKeys

    def __call__(self, varName, sectionName, layerNum):
        keyPrefix = f"{varName}/{sectionName}/fc{layerNum}/"
        matches = np.array([keyPrefix in key for key in self.allKeys])
        matchIndices = np.argwhere(matches).flatten()
        assert(len(matchIndices) == 1)
        matchKey = self.allKeys[matchIndices[0]]
        return matchKey


def logHist(data, bins, base):
    logData = np.log10(np.array(data) + base)
    counts, logBins = np.histogram(logData, bins=bins)
    return counts, 10 ** logBins


def syncLimits(axs):
    xlims = sum([list(ax.get_xlim()) for ax in axs], [])
    xlim = (min(xlims), max(xlims))
    ylims = sum([list(ax.get_ylim()) for ax in axs], [])
    ylim = (min(ylims), max(ylims))
    newLimits = [(ax.set_xlim(xlim), ax.set_ylim(ylim)) for ax in axs]
    return newLimits


class PlotHist:
    def __init__(self, useAbs, useLog, histBase, bins):
        self.useAbs = useAbs
        self.useLog = useLog
        self.histBase = histBase
        self.bins = bins

    def __call__(self, ax, rawData):
        if self.useLog:
            ax.set_xscale("log")
        data = (np.abs(rawData) if self.useAbs else rawData) + (self.histBase if self.useLog else 0)
        counts, _ = np.histogram(data, bins=self.bins)
        ax.hist(self.bins[:-1], bins=self.bins, weights=counts / np.sum(counts))
        ax.text(0.5, 1, f"$\mu=${np.mean(data):.1E} [{np.min(data):.1E}, {np.max(data):.1E}]",
                ha='center', va='top', transform=ax.transAxes, fontdict={'size': 8})


class PlotBars:
    def __init__(self, useAbs, useLog):
        self.useAbs = useAbs
        self.useLog = useLog

    def __call__(self, ax, means, stds, mins, maxes, labels):
        if self.useLog:
            ax.set_yscale('log')
        numLayers = len(labels)
        ax.plot(range(numLayers), means, 'or', label="$\mu$")
        ax.errorbar(range(numLayers), means, stds, label="$\sigma$", fmt='.', markersize=0, ecolor="black", lw=4)
        ax.errorbar(range(numLayers), means, [means - mins, maxes - means], label="range", fmt=".", markersize=0, ecolor="grey", lw=2)
        # ax.legend()
        ax.set_xticks(range(numLayers))
        ax.set_xticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
