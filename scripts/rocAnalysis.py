"""
A set of tools to draw and analyze Receiver Operating Characteristic Curves.
Description of method can be found here: Akobeng, Understanding diagnostic
tests 3: receiver operating characteristic curves, 2007

Required: a set of files containing results from BLAST runs with different
parameters, and different levels of sequence identity, (currently: 100, 99,
95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0)
where parameters are in the filename.
The id of the sequences must be of the form xx|yy|sequence identity level.
"""

from scipy import integrate
import operator
import copy

from dark import conversion



def _records(blastFilename):
    reader = conversion.JSONRecordsReader(blastFilename)
    for record in reader.records():
        yield record


def countHits(blastFile, cutoff):
    """
    Counts the number of reads that hit in each file at each level of sequence identity.

    @param blastFile: a file with blast output.
    @param cutoff: a bit score cutoff, reads below that will not be considered.
    @return: the name of the blastFile and a dictionary, with the level of sequence
        identity as key and the number of reads with that sequence identity as values
    """
    readsPerLevel = {'100': 0, '99': 0, '95': 0, '90': 0, '85': 0, '80': 0, '75': 0,
                     '70': 0, '65': 0, '60': 0, '55': 0, '50': 0, '45': 0, '40': 0,
                     '35': 0, '30': 0, '25': 0, '20': 0, '15': 0, '10': 0, '5': 0, '0': 0}
    t = blastFile.split('.')[2]
    title = t.split('/')[4]
    records = _records(blastFile)
    for record in records:
        query = record.query
        level = query.split('|')[2]
        try:
            alignment = record.alignments[0]
        except IndexError:
            continue
        score = alignment.hsps[0].bits
        if score > cutoff:
            readsPerLevel[level] += 1

    return title, readsPerLevel


# after the above function has been called, all dictionaries must be added into a large dictionary
# to be fed into calculateFrequencies


def calculateFrequencies(hitDict):
    """
    Calculates true negatives, true positives, false positives,
    false negatives, specificity, false positive rate and true positive rate.
    
    @param readsPerLevel: a dictionary of the form: blastn4-545 {'60': 56, '65': 115, ...}

    @return: a dictionary containing the scoringScheme as keys, and dictionaries for
        tn, tp, fp, fn, spec, fpr, tpr.
    """
    # transform the dict for each scoring scheme into a list
    listDict = {}
    for scoringScheme in hitDict:
        l = []
        for nr in [100, 99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]:
            l.append(hitDict[scoringScheme][str(nr)])
            listDict[scoringScheme] = l
    
    # calculate frequencies
    frequencies = {}
    
    for scoringScheme in listDict:
        tps = []
        tns = []
        fps = []
        fns = []
        fprs = []
        tprs = []
        spcs = []
        res = listDict[scoringScheme]
        for i, level in enumerate(res):
            tp = sum(res[:i+1])
            fp = sum(res[i+1:])
            fn = (594 * (i+1)) - tp
            tn = 594 * (22 - (i+1)) - fp
            tpr = float(tp)/float(tp+fn)
            try:
                fpr = float(fp)/float(fp+tn)
                spc = float(tn)/float(fp+tn)
            except ZeroDivisionError:
                fpr = 0
                spc = 0
            fprs.append(fpr)
            tprs.append(tpr)
            spcs.append(spc)
            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            
        frequencies[scoringScheme] = {'fprs': fprs, 'tprs': tprs, 'spcs': spcs,
                                      'tps': tps, 'tns': tns, 'fps': fps,
                                      'fns': fns, 'raw': res}
    return frequencies


def youdenIndex(frequencies):
    """
    Calculate the youden index.
    This is a measure for which cutoff should be used. Assumes that the optimal cutoff point
    is the point on the curve furthest away from the chance line going from (0, 0) to (1, 1).
    Formula: max{sensitivity + specificity - 1}

    @param frequencies: frequencies dictionary as returned from calculateFrequencies

    @return: a list of tuples with the coordinates of the point on each roc curve corresponding
        to the cutoff and a list of cutoffs.
    """
    level = [100, 99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
    youdenCoords = []
    cutoffs = []
    for scoringScheme in frequencies:
        youdens = []
        x = frequencies[scoringScheme]['spcs']
        y = frequencies[scoringScheme]['tprs']
        for i, l in enumerate(x):
            youden = x[i] + y[i] - 1
            youdens.append(youden)
        youdenIndex = max(youdens)
        cutoff = youdens.index(youdenIndex)
        element = (y[youdens.index(youdenIndex)], frequencies[scoringScheme]['fprs'][youdens.index(youdenIndex)])
        cutoffs.append(cutoff)
        youdenCoords.append(element)

    return youdenCoords, cutoffs


def f1Index(frequencies):
    """
    Calculate the F1 index.
    This is a measure for which cutoff should be used. Assumes that the optimal cutoff point
    is the point on the curve closest to (0, 1).
    Formula: min{(1-sensitivity)**2 + (1-specificity)**2}

    @param frequencies: frequencies dictionary as returned from calculateFrequencies
    """
    level = [100, 99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
    cutoffs = []
    f1Coords = []
    for scoringScheme in frequencies:
        f1s = []
        x = frequencies[scoringScheme]['spcs']
        y = frequencies[scoringScheme]['tprs']
        for i, l in enumerate(x):
            f1 = (1-x[i])**2 + (1-y[i])**2
            f1s.append(f1)
        f1Index = min(f1s)
        cutoff = f1s.index(f1Index)
        element = (y[f1s.index(f1Index)], frequencies[scoringScheme]['fprs'][f1s.index(f1Index)])
        cutoffs.append(cutoff)
        f1Coords.append(element)

    return f1Coords, cutoffs


def areaUnderCurve(frequencies):
    """
    Calculate the area under each curve.
    This acts as a measure for how good the test is at distinguishing between tp and tn.
    The larger the area under the curve (the closer to 1), the better the test. The closer
    the area to 0.5 the more rubbish is the test.

    @param frequencies: frequencies dictionary as returned from calculateFrequencies
    """
    freqs = copy.copy(frequencies)
    integrated = {}
    for scoringScheme in freqs:
        x = freqs[scoringScheme]['fprs']
        y = freqs[scoringScheme]['tprs']
        # normalize curves
        x.append(0.0)
        y.append(0.0)
        x.insert(0, 1.0)
        y.insert(0, 1.0)
        # integrate
        y_int = integrate.trapz(y, x)
        integrated[scoringScheme] = y_int * -1

    return integrated


def areaUnderCurveIndex(integrated):
    """
    Returns a dictionary where the key is a scoringScheme, and the value goes from 0, 1, 2...
    in the order of area under curve.

    @param integrated: result from areaUnderCurve
    """
    sortedIntegrated = sorted(integrated.iteritems(), key=operator.itemgetter(1))[::-1]

    indexes = {}
    for i, item in enumerate(sortedIntegrated):
        indexes[item[0]] = i

    return indexes
