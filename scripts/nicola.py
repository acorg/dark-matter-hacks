"""
This file contains functions to analyse some BLAST runs for Nicola.
Workflow:
BLAST all of Nicolas sequences against a database with all avian flu sequences.
For each of Nicolas sequence, make a dot plot, showing percent sequence
identity on the Y-axis and all titles that match against the sequence
sorted by percent sequence identity on the X-axis. Dots can be coloured
according to species or location of the subject.
"""


import matplotlib.pylab as plt
import numpy as np
import re
from scipy.cluster.vq import kmeans, vq
from scipy import stats
from Bio import SeqIO
from collections import defaultdict
from itertools import cycle

from dark import conversion
from dark.dimension import dimensionalIterator
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation

# regexes and lists for coloring
# colour by all taxonomic groups
ANSERIFORMESREGEX = re.compile('duck|waterfowl|widgeon|wigeon|pintail|'
                               'garganey|teal|mallard|shoveler|gadwall|'
                               'canvasback|redhead|pochard|scaup|merganser|'
                               'eider|scoter|goldeneye|bufflehead|smew|'
                               'goosander|shelduck|swan|goose|coot|'
                               'scooter', re.I)

CHARADRIIFORMESREGEX = re.compile('gull|tern|shorebird|turnstone|sandpiper|'
                                  'dunlin|watercock|lapwing|knot|curlew|'
                                  'kittiwake|snipe|sanderling|stint|plover|'
                                  'woodcock|murre|puffin|guillemot|shearwater|'
                                  'oystercatcher', re.I)

PELICANIFORMESREGEX = re.compile('pelican|heron|egret|cormorant|grebe|crane',
                                 re.I)

DOMESTICREGEX = re.compile('chicken|FPV|turkey|quail|guinea|ostrich|emu|fowl|'
                           'pheasant|poultry|rhea|chicken|partridge|bantam|'
                           'perdix|peacock|chukkar', re.I)

COLUMBIFORMESREGEX = re.compile('pigeon|dove', re.I)

PASSERIFORMESREGEX = re.compile('sparrow|wheatear|starling|munia|finch|'
                                'softbill|shrike|robin|blackbird|myna|iora|'
                                'magpie|crow', re.I)

PSITTATICIFORMESREGEX = re.compile('budgerigar|parrakeet|psittacine|cockatoo|'
                                   'conure|macaw|parakeet|parrot', re.I)

RAPTORREGEX = re.compile('hawk|kestrel|goshawk|buzzard|falcon|owl|eagle|'
                         'harrier', re.I)

INDETERMINATEHOSTREGEX = re.compile('avian|aquatic|bird', re.I)

NEITHER = 'grey'
CHARADRIIFORMES = 'red'
ANSERIFORMES = 'green'
PELICANIFORMES = 'cyan'
DOMESTIC = 'blue'
COLUMBIFORMES = 'magenta'
PASSERIFORMES = 'orange'
PSITTATICIFORMES = '#4C0B5F'
RAPTOR = 'yellow'
INDETERMINATEHOST = 'black'

# colour by Anseriformes and Charadriiformes
# Anseriformes
ANATINAEREGEX = re.compile('waterfowl|wigeon|widgeon|pintail|garganey|teal|'
                           'mallard|shoveler|gadwall|black duck|scooter', re.I)

AYTHYAREGEX = re.compile('canvasback|tufted duck|redhead|pochard|'
                         'ferroginous duck|scaup', re.I)

MERGINIREGEX = re.compile('merganser|eider|scoter|long-tailed duck|goldeneye|'
                          'bufflehead|smew|goosander', re.I)

TADORNINIREGEX = re.compile('shelduck', re.I)

SWANREGEX = re.compile('swan', re.I)

GOOSEREGEX = re.compile('goose', re.I)

# Charadriiformes

GULLREGEX = re.compile('gull', re.I)

TERNREGEX = re.compile('tern|Arctic tern|Common Tern|little tern|sooty tern',
                       re.I)

WADERREGEX = re.compile('shorebird|ruddy turnstone|turnstone|sandpiper|dunlin|'
                        'watercock|northern lapwing|knot|curlew|sanderling|'
                        'stint|plover|woodcock', re.I)

AUKREGEX = re.compile('murre|puffin|guillemot|shearwater', re.I)

DOMESTICREGEX = re.compile('chicken|FPV|turkey|quail|guinea|ostrich|emu|fowl|'
                           'pheasant|poultry|rhea|chicken|partridge|bantam|'
                           'perdix|peacock|chukkar', re.I)

ANATINAE = 'green'
AYTHYA = 'blue'
MERGINI = 'cyan'
TADORNINI = '#00FF00'  # light green
SWAN = '#088A68'  # sea green
GOOSE = '#A9F5D0'  # light sea green
GULL = 'red'
TERN = 'orange'
WADER = '#4C0B5F'  # violett
AUK = 'yellow'
DOMESTIC = 'black'
OTHER = 'grey'


# countries
EURASIAREGEX = ['Republic of Georgia', 'Sweden', 'Norway', 'Hokkaido',
                'Mongolia', 'Iceland', 'Astrakhan', 'Siberia',
                'Netherlands', 'Potsdam', 'Ukraine', 'Hubei', 'Hong Kong',
                'England', 'Czech Republic', 'United Kingdom',
                'Czechoslovakia', 'Shimane', 'Germany', 'Leipzig',
                'Scotland', 'Henan', 'Stralsund', 'HuNan', 'Marquenterre',
                'Berlin', 'Rugen', 'Jena', 'Korea', 'Wagun', 'Postdam',
                'France', 'Italy', 'Dongting', 'Heinersdorf', 'Altai',
                'Egypt', 'Guangxi', 'Taiwan', 'Gurjev', 'Jiangxi', 'HK',
                'Chabarovsk', 'Shantou', 'England-Q', 'South Africa',
                'Jiang Xi', 'Nanjing', 'Vietnam', 'Primorie', 'India',
                'Denmark', 'Thailand', 'Guangdong', 'Xianghai', 'Spain',
                'Lebanon', 'Shiga', 'Jiangsu', 'Bavaria', 'Hunan',
                'Eastern China', 'PT', 'Belgium', 'Australia', 'Slovenia',
                'Yunnan', 'Yokohama', 'Guizhou', 'Fujian', 'Singapore-Q',
                'Hungary', 'Iran', 'Japan', 'Pakistan', 'Moscow', 'Israel',
                'Shimone', 'Nanchang', 'Saudi Arabia', 'Hampshire', 'Perak',
                'Gloucestershire', 'SanJiang', 'Hei Longjiang', 'Ireland',
                'Dubai', 'Netherlands2', 'Zhejiang', 'Malaysia', 'Sanjiang',
                'Chany', 'Portugal', 'Heilongjiang', 'Qinghai', 'AUS',
                'Tasmania', 'Western Australia', 'HONG KONG', 'Hebei',
                'HuBei', 'Beijing', 'Victoria', 'ZhaLong', 'Yan chen',
                'Shandong', 'Buryatiya', 'Interior Alaska', 'Xuyi', 'Hongze',
                'Laos', 'Mongolia', 'Osaka', 'Yangzhou', 'Aktau', 'Chiba',
                'Huadong', 'Shanghai', 'Anyang', 'Akita', 'Gui Yang',
                'Karachi', 'Tyva', 'Jilin', 'New Zealand', 'Lao',
                'Suphanburi', 'Bulgaria', 'Aomori', 'Fukushima', 'Hamanaka',
                'Yamaguchi', 'Tochigi', 'Kagoshima', 'Anhui', 'Singapore',
                'Guiyang', 'Nam Dinh', 'Ninh Binh', 'Sichuan', 'Phu Tho',
                'Oita', 'Son La', 'Thai Ninh', 'Thai Binh', 'Ha Nam',
                'Nghe An', 'Hai Duong', 'Ha Tinh', 'Bac Gieng', 'Bac Giang',
                'Mong Cai', 'Primorje', 'Cao Bang', 'Bac Ninh', 'Kyoto',
                'Toyama', 'Aichi', 'Oregon', 'Switzerland', 'Pavia', 'Austria',
                'NZL', 'Northern Ireland', 'Finland', 'Viet Nam', 'KOREA',
                'Newcastle', 'Cheongju', 'Morpeth', 'Middle East', 'Jordan',
                'Neve Ilan', 'Beit HaLevi', 'Givat Haim', 'Eshkol',
                'Tel Adashim', 'Talmei Elazar', 'Narita', 'Tianjing',
                'China', 'Gansu', 'Shijiazhuang', 'Shenzhen', 'Heibei',
                'Emirates', 'Gangxi', 'Xuzhou', 'Atyrau', 'Volga',
                'Turkmenistan', 'Novosibirsk', 'Almaty', 'Miyagi', 'Tsukuba',
                'Praimoric', 'Crimea', 'Italy12rs206-2', 'Wales', 'Iwate',
                'Gunma', 'Fukui', 'Seongdong', 'bavaria', 'Indonesia',
                'Netherland-18', 'Barrow', 'Tainan', 'San Jiang', 'Changhua',
                'Geumgang', 'Wenzhou', 'Rizhao', 'Korgalzhyn', 'Saitama',
                'GuangXi', 'Bhutan', 'Bunbury', 'Perth', 'Czech', 'Nha Trang',
                'PoyangLake', 'Poyang Lake', 'Buryatuya', 'Ningxia',
                'Guangzhou', 'Dutch', 'Dobson', 'england', 'Brescia',
                'Turkmenia', 'Rostock', 'Niigata', 'Shizuoka', 'Chakwal',
                'Rawalpindi', 'Murree', 'Mansehra', 'Abbottabad',
                'Queensland', 'HongKong', 'It', 'Burytiya', 'Ibaraki',
                'Weybridge|Segment:6|Subtype:H7N7|Host:Avian',
                'Mangystau', 'Ulster', 'Miyagil', 'Burjatia', 'kumamoto',
                'Tunka', 'NSW', 'New South Wales', 'Sheny', 'Liaoning',
                'Tibet', 'Shanxi', 'Islamabad', 'Lahore-Pakistan',
                'West Bengal', 'Manshera', 'SouthKalimantan', 'Palopo',
                'EastKalimantan', 'Islamabad-Pakistan', 'Mansehra-Pakistan',
                'Pakistan-Khi', 'Sihala', 'Pakistan-Lhr',
                'Afghanistan', 'Deli Serdang', 'Suzdalka', 'Chakshahzad',
                'Croatia', 'Omsk', 'Krasnoozerskoe', 'Taput', 'Siantar',
                'Pinrang', 'Pare Pare', 'BIH', 'Denpasar', 'Tanjung Balai',
                'Tana Toraja', 'Simalungun', 'Medan', 'Maros', 'Mamuju',
                'Karo', 'Gianyar', 'Aceh Barat', 'South Kalimantan',
                'Langkat', 'Bangladesh', 'Suzhou', 'Bantaeng', 'Xinjiang',
                'East Java', 'West Java', 'Sidrap', 'Bantul', 'Ciamis',
                'Bireun', 'Aceh Besar', 'EastJava', 'CentralJava',
                'Tabanan', 'Pidie', 'Banten', 'Heuwiese', 'Astr', 'Jalisco',
                'Slovakia', 'Koymor', 'Yangcheng', 'Tagarskoe lake',
                'South Kazakhstan', 'Ust-Ilimsk', 'Gaoyou',
                'Dobson|Segment:8|Subtype:H7N7|Host:Avian', 'FPV',
                'Weybridge|Segment:8|Subtype:H7N7|Host:Avian',
                'Bethlehem-Glilit', 'Gippsland']

NORTHAMERICAREGEX = ['Southcentral Alaska', 'Southeastern Alaska', 'Quebec',
                     'New Brunswick', 'DE', 'Newfoundland', 'Maryland',
                     'Minnesota', 'NJ', 'Alaska', 'California',
                     'Delaware', 'Georgia', 'Delaware Bay', 'New Jersey',
                     'New York', 'MT', 'Alberta', 'NY', 'PA', 'Pennsylvania',
                     'CA', 'Washington', 'ST', 'MN', 'Wisconsin', 'TX',
                     'Texas', 'Stockton', 'Missouri', 'ALB', 'Ohio',
                     'Virginia', 'Arkansas', 'Iowa', 'North Carolina',
                     'Nova Scotia', 'Tennessee', 'Illinois', 'New Mexico',
                     'Mississippi', 'Canada', 'Alabama', 'Louisiana',
                     'Arizona', 'LA', 'Ontario', 'Massachusetts', 'OH',
                     'Connecticut', 'MD', 'Colorado', 'Memphis',
                     'Saskatchewan', 'North Dakota', 'VA', 'Michigan',
                     'Maine', 'Prince Edward Island', 'JX', 'WI',
                     'MO', 'SD', 'IN', 'South Dakota',
                     'Newfoundland and Labrador', 'Utah', 'Kentucky',
                     'CO', 'Massachussetts', 'Indiana', 'Maryland', 'QC',
                     'SK', 'Montana', 'GA', 'Manitoba', 'Edmonton', 'Nunavet',
                     'British Columbia', 'NB', 'BC', 'Barbados',
                     'Wyoming', 'NV']

SOUTHAMERICAREGEX = ['Chile', 'Argentina', 'Guatemala', 'Brazil',
                     'Mexico', 'Ilha de Canelas']

AFRICAREGEX = ['Zambia', 'Zimbabwe', 'SouthAfrica', 'Nigeria']

EURASIA = '#E0F2F7'  # blue
NORTHAMERICA = '#FBEFEF'  # red
SOUTHAMERICA = '#F5F6CE'  # yellow
AFRICA = '#CEF6D8'  # green


# General utility functions

def _getCountry(title):
    """
    Finds out from which continent a title of a sequence originates.

    @param title: The title of a blastHit.

    @return: A C{int} denoting Continent
    """
    try:
        country = title.split('/')[2]
    except IndexError:
        return 0

    if country in EURASIAREGEX:
        return EURASIA
    elif country in NORTHAMERICAREGEX:
        return NORTHAMERICA
    elif country in SOUTHAMERICAREGEX:
        return SOUTHAMERICA
    elif country in AFRICAREGEX:
        return AFRICA
    else:
        return 'white'


def _getBird(title, colorBy='all'):
    """
    Finds out what colour should be assigned to a title, given
    what bird it is.

    @param title: The title of a blastHit.
    @param colorBy: How the coloring should be done. Either color
        by all taxonomic groups ('all') or color by anseriformes
        and charadriiformes ('anseriformes and charadriiformes').
    """
    if colorBy == 'anseriformes and charadriiformes':
        anatinae = ANATINAEREGEX.search(title)
        aythya = AYTHYAREGEX.search(title)
        mergini = MERGINIREGEX.search(title)
        tadornini = TADORNINIREGEX.search(title)
        swan = SWANREGEX.search(title)
        goose = GOOSEREGEX.search(title)
        gull = GULLREGEX.search(title)
        tern = TERNREGEX.search(title)
        wader = WADERREGEX.search(title)
        auk = AUKREGEX.search(title)
        domestic = DOMESTICREGEX.search(title)

        if anatinae:
            return ANATINAE
        elif aythya:
            return AYTHYA
        elif mergini:
            return MERGINI
        elif tadornini:
            return TADORNINI
        elif swan:
            return SWAN
        elif goose:
            return GOOSE
        elif gull:
            return GULL
        elif tern:
            return TERN
        elif wader:
            return WADER
        elif auk:
            return AUK
        elif domestic:
            return DOMESTIC
        else:
            return OTHER

    elif colorBy == 'all':
        anseriformes = ANSERIFORMESREGEX.search(title)
        charadriiformes = CHARADRIIFORMESREGEX.search(title)
        pelicaniformes = PELICANIFORMESREGEX.search(title)
        domestic = DOMESTICREGEX.search(title)
        columbiformes = COLUMBIFORMESREGEX.search(title)
        passeriformes = PASSERIFORMESREGEX.search(title)
        psittaticiformes = PSITTATICIFORMESREGEX.search(title)
        raptor = RAPTORREGEX.search(title)
        indetermined = INDETERMINATEHOSTREGEX.search(title)

        if anseriformes:
            return ANSERIFORMES
        elif charadriiformes:
            return CHARADRIIFORMES
        elif pelicaniformes:
            return PELICANIFORMES
        elif domestic:
            return DOMESTIC
        elif columbiformes:
            return COLUMBIFORMES
        elif passeriformes:
            return PASSERIFORMES
        elif psittaticiformes:
            return PSITTATICIFORMES
        elif raptor:
            return RAPTOR
        elif indetermined:
            return INDETERMINATEHOST
        else:
            return NEITHER


def computePercentId(alignment):
    """
    Calculates the percent sequence identity of an alignment.

    @param alignment: An instance of C{Bio.Blast.Record.Blast.Alignment}

    @return: A C{float} with the percent identity
    """
    query = alignment.hsps[0].query
    sbjct = alignment.hsps[0].sbjct
    length = len(query)

    identical = 0
    for queryBase, subjectBase in zip(query, sbjct):
        identical += int(queryBase == subjectBase)

    identity = identical / float(length) * 100
    return identity


def _records(blastFilename):
    """
    Generate blast records from a json file.
    """
    reader = conversion.JSONRecordsReader(blastFilename)
    for record in reader.records():
        yield record


# functions for working with distance graphs

def distancePlot(record, distance='bit', colorBy='all', continents=True,
                 imageFile=False, createFigure=True, showFigure=False,
                 readsAx=False):
    """
    Produces a rectangular panel of graphs that each show sorted distances for
    a read. Read hits against a certain strain (see find, below) are
    highlighted.

    @param record: A C{readAlingments} instance.
    @param distance: The measure of distance read out from the blastFile,
        either 'bit' of 'percentId'.
    @param colorBy: How the coloring should be done. Either color
        by all taxonomic groups ('all') or color by anseriformes
        and charadriiformes ('anseriformes and charadriiformes').
    @param continents: if C{bool} True, color the backgound of each title by
        which continent the title is from.
    @param imageFile: a C{string} filename where the figure should be saved to.
    @param readsAx: If not None, use this as the subplot for displaying reads.

    @return: Returns the largest distance, and the number of distances that
        were plotted.
    """
    alignments = record.alignments
    title = str(record.query)

    fig = plt.figure(figsize=(30, 10))
    ax = readsAx or fig.add_subplot(111)

    if distance != 'bit':
        for alignment in alignments:
            distance = computePercentId(alignment)
            alignment.hsps[0].bits = distance

    sortedAlignments = sorted(alignments,
                              key=lambda k: k.hsps[0].bits,
                              reverse=True)

    distances = []
    titles = []

    for alignment in sortedAlignments:
        distances.append(alignment.hsps[0].bits)
        titles.append(alignment.title)

    x = np.arange(0, len(distances))

    # plot black line with distance
    ax.plot(x, distances, 'k', linewidth=0.5)
    ax.yaxis.grid(linewidth=0.1, color='k', linestyle='--')
    for item in np.arange(0, len(distances), 10):
        plt.axvline(item, linewidth=0.1, color='k', linestyle='--')
    plt.title(title + '\n', fontsize=20)
    plt.ylabel('Bit scores' if distance == 'bit' else '% id', fontsize=15)
    # plot green and red dots denoting gulls and ducks
    for i, alignment in enumerate(sortedAlignments):
        title = alignment.title
        y = distances[i]
        bird = _getBird(title, colorBy=colorBy)
        ax.plot([i], [y], markerfacecolor=bird, marker='o', markersize=3,
                markeredgecolor=bird)
    plt.xticks(np.arange(0, len(distances), 1.0), fontsize=4)
    #if not readsAx:
    titlesToPlot = []
    for title in titles:
        splitted = title.split('|')
        titlesToPlot.append(splitted[1][26:] + '/' + splitted[3][8:])
    ax.set_xticklabels(titlesToPlot, rotation=270, fontsize=4)

    if continents:
        for i, shortTitle in enumerate(titlesToPlot):
            country = _getCountry(shortTitle)
            ax.axvspan(i-0.5, i+0.5, color=country, linewidth=0.5)

    if createFigure:
        if showFigure:
            plt.show()
        if imageFile:
            fig.savefig(imageFile, bbox_inches='tight')

    return distances[0], len(distances)


def distancePanel(blastName, matrix, distance='bit', colorBy='all',
                  continents=True, outputDir=False):
    """
    Make a panel of distance plots generated with the distancePlot
    function above.

    @param blastName: File with blast output
    @param matrix: A matrix of strings corresponding to record.queries
        at the position where the plot of a given record should be.
    @param distance: The measure of distance read out from the blastFile,
        either 'bit' or 'percentId'.
    @param colorBy: How the coloring should be done. Either color
        by all taxonomic groups ('all') or color by anseriformes
        and charadriiformes ('anseriformes and charadriiformes').
    @param continents: if C{bool} True, color the backgound of each title by
        which continent the title is from.
    @param outputDir: if not C{bool} false,a C{str} of where the
        individual panels should be written to.
    """
    cols = 8
    rows = 53
    figure, ax = plt.subplots(rows, cols, squeeze=False)
    maxDistance = 0
    maxReads = 0
    allRecords = _records(blastName)
    count = 0
    for record in allRecords:
        query = record.query
        print count, query
        try:
            coordinates = matrix[query]
            row = coordinates[0]
            col = coordinates[1]
        except KeyError:
            # if that record is not present in matrix, leave it out.
            continue
        if outputDir:
            figureTitle = '%s%d-%d.svg' % (outputDir, row, col)
            localMaxDistance, numberOfReads = distancePlot(
                record, colorBy=colorBy, continents=continents,
                distance=distance, showFigure=False, createFigure=False,
                imageFile=figureTitle, readsAx=None)
        else:
            localMaxDistance, numberOfReads = distancePlot(
                record, colorBy=colorBy, continents=continents,
                distance=distance, showFigure=False, createFigure=True,
                imageFile=False, readsAx=ax[row][col])

        count += 1
        try:
            title = query.split('(')[1]
            subtype = query.split('(')[2][:-2]
            segment = query.split(' ')[0]
        except IndexError:
            # this is for one title which doesn't fit the usual format
            segment = 'Segment 2'
            subtype = 'H3N8'
            title = query
        ax[row][col].set_title('%s, %d, %s \n %s' % (
                               segment, numberOfReads,
                               subtype, title), fontsize=10)
        if localMaxDistance > maxDistance:
            maxDistance = localMaxDistance
        if numberOfReads > maxReads:
            maxReads = numberOfReads

    coords = dimensionalIterator((rows, cols))

    allRecords = _records(blastName)

    # normalize plots
    for i, record in enumerate(matrix):
        row, col = coords.next()
        a = ax[row][col]
        a.set_ylim([0, maxDistance + 1])
        a.set_xlim([0, maxReads + 1])
        a.set_yticks([])
        a.set_xticks([])

    for row, col in coords:
        ax[row][col].axis('off')

    plt.subplots_adjust(hspace=0.4)
    figure.suptitle('X: 0 to %d, Y (%s): 0 to %d' %
                    (maxReads, ('Bit scores' if distance == 'bit'
                                else '% id'),
                     maxDistance), fontsize=20)
    figure.set_size_inches(5 * cols, 3 * rows, forward=True)
    figure.show()


# ===================================================
# THE CODE FROM DOWN HERE IS DEGRADED AND NOT TESTED!
# ===================================================

def makeListOfHitTitles(blastName):
    """
    Makes a list of titles that are hit.

    @param blastName: File with blast output.

    @return: A list of titles.
    """
    blastRecords = blast.BlastRecords(blastName)
    interesting = blastRecords.filterHits(withBitScoreBetterThan=50)
    titlesList = interesting.titles.keys()

    return titlesList


def heatMapFromPanel(blastName, matrix):
    """
    Each plot in the panel above is assigned a value, based on a
    metric to count inversions. Plot those values as a heatmap.

    @param blastName: File with blast output
    @param matrix: A matrix of strings corresponding to record.queries
        at the position where the plot of a given record should be.

    @return: The array that was plotted
    """
    cols = 8
    rows = 53
    array = ([[0 for _ in range(cols)] for _ in range(rows)])

    blastRecords = blast.BlastRecords(blastName)
    records = blastRecords.records()

    for record in records:
        query = record.query
        queryType = _getBird(query)
        if queryType != NEITHER:
            try:
                coordinates = matrix[query]
                row = coordinates[0]
                col = coordinates[1]
            except TypeError:
                # if that record is not present in matrix, leave it out.
                continue
            alignments = record.alignments
            for alignment in alignments:
                distance = computePercentId(alignment)
                alignment.hsps[0].bits = distance
            sortedAlignments = sorted(alignments,
                                      key=lambda k: k.hsps[0].bits,
                                      reverse=True)

            for i, alignment in enumerate(sortedAlignments):
                titleType = _getBird(alignment.title)
                if titleType != NEITHER:
                    if queryType != titleType:
                        array[row][col] = i
                        #print i,
                        break

        else:
            print query

    # plot the generated matrix:
    fig = plt.figure(1, figsize=(10, 20))
    ax = fig.add_subplot(111)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # other color schemes that may be useful: PiYG, Spectral
    heatMap = ax.imshow(array, interpolation='nearest', cmap='hot')
    for i, segment in enumerate(['HA', 'NA', 'PB2', 'PB1',
                                 'PA', 'NP', 'MP', 'NS']):
        ax.text(i, -1.5, segment, horizontalalignment='center',
                size='medium', color='k', rotation=45)

    # plot the labels and colorbar
    for i, row in enumerate(matrix):
        title = row[2]
        splittedTitle = title.split('(')
        try:
            tick = splittedTitle[1] + '(' + splittedTitle[2][:-1]
        except IndexError:
            title = row[4]
            splittedTitle = title.split('(')
            try:
                tick = splittedTitle[1] + '(' + splittedTitle[2][:-1]
            except IndexError:
                tick = 'A/Anas platyrhynchos/Belgium/12827/2007(H3N8)'
        ax.text(18, i, tick, horizontalalignment='center', size='medium',
                color='k')
    fig.colorbar(heatMap, shrink=0.25, orientation='vertical')

    return array


# Functions for working with distance matrix

def makeDistanceMatrix(blastName, fastaName, titlesList=None,
                       masked=False, toFile=False, addTitles=False,
                       missingValue=0.0, distance='bit'):
    """
    Takes a blast output file, returns a distance matrix.

    @param blastName: File with blast output
    @param fastaName: A fastafile with the titles that were blasted,
        in the order that it should be in the matrix. Or a list of
        titles that were blasted, in the order that they should be
        in the matrix.
    @param titlesList: If not C{False}, a list of titles, in the order
        that they should be in the matrix.
    @param masked: If C{True}, the matrix returned will have cells with no hits
        masked out.
    @param toFile: If not C{False}, a C{str} file name where thematrix should
        be written to.
    @param addTitles: If C{True} the titles of hits and fasta sequences will be
        added to the matrix.
    @param missingValue: The value used in the matrix if no distance is given.
    @param distance: The measure of distance read out from the blastFile,
        either 'bit' or 'percentId'.

    @return: the distance matrix that was made, the titlesList, and a C{list}
        of sequence titles.

    NOTE: masked = True and missingValue != 0 does not work!
    """
    if not titlesList:
        titlesList = makeListOfHitTitles(blastName)
    titlesDict = dict((title, index)
                      for (index, title) in enumerate(titlesList))

    if type(fastaName) == list:
        fastaList = fastaName
    else:
        fastaList = list(record.description for record
                         in SeqIO.parse(fastaName, 'fasta'))
    fastaDict = {item: index for (index, item) in enumerate(fastaList)}

    if not masked:
        initMatrix = [[missingValue for _ in range(
                      len(titlesList))] for _ in range(len(fastaList))]
    else:
        initMatrix = np.mp.masked_array(
            [[missingValue for _ in range(
                len(titlesList))] for _ in range(len(fastaList))])

    BlastRecords = blast.BlastRecords(blastName)
    records = BlastRecords.records(bitScoreCutoff=50)

    for record in records:
        query = record.query
        # get position of query in queryList
        try:
            queryIndex = fastaDict[query]
        except ValueError:
            # if query not present in fastaDict, continue
            continue
        for alignment in record.alignments:
            title = alignment.title
            if distance == 'bit':
                dist = alignment.hsps[0].bits
            else:
                dist = computePercentId(alignment)
            # get position of title in titlesList
            try:
                subjectIndex = titlesDict[title]
            except ValueError:
                # if query not present in titlesDict, continue
                continue
            # add distance to matrix
            initMatrix[queryIndex][subjectIndex] = dist

    # mask values that are 0 (those that were not hit)
    if masked:
        initMatrix = np.ma.masked_array(initMatrix, mask=(initMatrix < 1))

    if addTitles:
        initMatrix = distanceMatrixWithBorders(initMatrix,
                                               titlesList, fastaList)

    if toFile:
        matrixToFile(toFile, initMatrix)

    return initMatrix, titlesList, fastaList


def distanceMatrixWithBorders(matrix, titlesList, fastaList):
    """
    Add titles to distance matrix.

    @param matrix: Matrix as returned from makeDistanceMatrix.
    @param titlesList: A list of titles, in the order that they
        should be in the matrix
    @param fastaList: List of titles that were blasted,
        in the order that they should be in the matrix

    @return: The altered distance matrix
    """
    #for matrixElement, fastaElement in zip(matrix, fastaList):
    #    matrixElement.insert(0, fastaElement)
    count = 0
    for item in matrix:
        item.insert(0, fastaList[count])
        count += 1

    # add an empty element to the beginning of titlesList.
    titlesList.insert(0, '')
    # insert titlesList as the first element of the matrix.
    matrix.insert(0, titlesList)

    return matrix


def matrixToFile(fileName, matrix):
    """
    Writes a matrix to a tab delimited file.

    @param fileName: The name of the file where the distance matrix
        should be written to.
    @param matrix: A distance matrix, as returned from makeDistanceMatrix.
    """
    first = True
    with open(fileName, 'w') as fp:
        for item in matrix:
            if first:
                first = False
                pass
            else:
                fp.write('\n')
            for nr in item:
                fp.write(str(nr) + ',')


def makeAffinityMatrix(matrix):
    """
    Make affinity matrix out of distance matrix.
    """
    sequenceTitles = []
    distanceMatrix = []
    sequenceColors = []

    for item in matrix[1:]:
        sequenceTitles.append(item[0])
        distanceMatrix.append(item[1:])

    for title in sequenceTitles:
        bird = _getBird(title)
        if bird == 1:
            sequenceColors.append('red')
        elif bird == 2:
            sequenceColors.append('blue')
        elif bird == 0:
            sequenceColors.append('black')

    distMat = np.array([row for row in distanceMatrix]).astype(np.float)

    affinityMatrix = np.array([100 - row for row in distMat]).astype(np.float)

    return affinityMatrix, sequenceColors, sequenceTitles


def agglomerativeHierarchical(matrix, k, linkage='ward'):
    """
    Performs hierarchical clustering.

    @param matrix: an affinity matrix, as returned from makeAffinityMatrix().
    @param k: number of clusters
    @param linkage: type of linkage ('ward', 'average', 'complete')
    """
    # perform clustering
    ward = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(matrix)
    label = ward.labels_
    # plotting
    plt.figure()
    for l in np.unique(label):
        plt.plot(matrix[label == l, 0], matrix[label == l, 1],
                 matrix[label == l, 2], 'o',
                 color=plt.cm.jet(np.float(l) / np.max(label + 1)))
    plt.title('agglomerative hierarchical clustering, linkage: %s, k: %d' % (
              linkage, k))
    plt.xlim(-10)
    plt.ylim(-10, 110)


def doAffinityPropagation(matrix, sequenceColors, sequenceNames):
    """
    Clustering with affinity propagation.

    @param matrix: an affinity matrix, as returned from makeAffinityMatrix().
    @param sequenceColors: a list of colors corresponding to whether the
        title of the row is a duck or a gull or neither. Returned from
        makeAffinityMatrix().
    """
    af = AffinityPropagation(affinity='precomputed').fit(matrix)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    # Plot result
    plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cycle('gcmykgcmykgcmykgcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = matrix[cluster_centers_indices[k]]
        # plot the individual sample dots
        plt.plot(matrix[class_members, 0], matrix[class_members, 1],
                 col + '.', markersize=15)
        # plot the centroids
        plt.plot(cluster_center[0], cluster_center[1], 'o',
                 markerfacecolor=col, markeredgecolor='k', markersize=14)
        # adds the starry lines
        for x in matrix[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
        # overplot each dot according to whether it's duck or gull
        for i, item in enumerate(matrix):
            color = sequenceColors[i]
            plt.plot(item[0], item[1], 'o', markersize=4,
                     markeredgecolor=color, markerfacecolor=color)

    plt.title('affinity propagation, clusters: %d' % n_clusters_)

    plt.show()

    clusternames = defaultdict(list)

    for i, label in enumerate(labels):
        clusternames[label].append(sequenceNames[i])

    # clusternames now holds a map from cluster label to list of sequence names
    # Print out the label with the list
    for k, v in clusternames.items():
        print k, v


def kMeansCluster(matrix, k, sequenceNames):
    """
    Takes a matrix, runs k-means clustering
    Adapted from http://glowingpython.blogspot.co.uk/
    2012/04/k-means-clustering-with-scipy.html

    @param matrix: A matrix (numpy array) for clustering (no borders).
    @param k: Number of clusters.

    @return: The computed centroids, indexes, and distortions.
    """
    # computing k-means
    centroids, kMeansDistortion = kmeans(matrix, k)

    # assign each sample to a cluster
    index, vqDistortion = vq(matrix, centroids)

    # plotting
    for i in range(k):
        plt.plot(matrix[index == i, 0], matrix[index == i, 1], 'bo')
             #matrix[index == 1, 0], matrix[index == 1, 1], 'ro')
        plt.plot(centroids[:, k], centroids[:, k], 'gs', markersize=8)
    plt.title('k-means, k: %s' % k)
    #plt.xlim(-10, 110)
    #plt.ylim(-10, 110)
    plt.show()

    clusternames = defaultdict(list)

    for i, ind in enumerate(index):
        clusternames[ind].append(sequenceNames[i])

    #clusternames now holds a map from cluster label to list of sequence names
    #Print out the label with the list
    for k, v in clusternames.items():
        print k, v

    return centroids, index, vqDistortion, kMeansDistortion


def _annotateTitles(titlesList):
    """
    Takes a list of titles, and returns a list of tuples,
    where each tuple is a title and 0, 1, 2, depending
    on whether the title is a gull or a duck.

    @param titlesList: A list of titles.

    @return: A C{list} of tuples, each tuple containing a title
        and a number indicating whether the title is a GULL, DUCK
        or NEITHER.
    """
    annotatedTitlesList = [(title, _getBird(title)) for title in titlesList]

    return annotatedTitlesList


def distancesBoxPlot(blastName, fastaName, plotTitle, distance='bit',
                     stat=True):
    """
    Draws a boxplot of the distances between ducks and gulls.

    @param blastName: File with blast output
    @param fastaName: A fastafile with the titles that was blasted,
        in the order that it should be in the matrix.
    @param plotTitle: A C{str} title of the plot
    @param stat: If C{True}, print a legend with the results of an
        unpaired t-test.

    @return: Four C{lists} of distances between ducks, gulls, duck-gulls
        and gull-ducks.
    """
    matrix, titlesList, fastaList = makeDistanceMatrix(blastName, fastaName,
                                                       missingValue=0,
                                                       distance=distance)

    annotatedTitlesList = _annotateTitles(titlesList)
    annotatedFastaList = _annotateTitles(fastaList)

    gullgullDistances = []
    duckduckDistances = []
    duckgullDistances = []
    gullduckDistances = []

    rows = len(matrix) - 1
    cols = len(matrix[0]) - 1
    for row in range(rows):
        for col in range(cols):
            # skip if no bird was assigned
            if annotatedFastaList[row][1] == NEITHER:
                continue
            #if matrix[row][col] != 0:
            if (annotatedFastaList[row][1] == GULL and
                    annotatedTitlesList[col][1] == GULL):
                gullgullDistances.append(matrix[row][col])
            elif (annotatedFastaList[row][1] == GULL and
                  annotatedTitlesList[col][1] == DUCK):
                gullduckDistances.append(matrix[row][col])
            elif (annotatedFastaList[row][1] == DUCK and
                  annotatedTitlesList[col][1] == DUCK):
                duckduckDistances.append(matrix[row][col])
            elif (annotatedFastaList[row][1] == DUCK and
                  annotatedTitlesList[col][1] == GULL):
                duckgullDistances.append(matrix[row][col])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = [gullgullDistances, duckduckDistances,
                 duckgullDistances, gullduckDistances]
    numBoxes = len(distances)
    medians = range(numBoxes)
    bp = ax.boxplot(distances)

    for i in range(numBoxes):
        med = bp['medians'][i]

        medianY = []
        for j in range(2):
            medianY.append(med.get_ydata()[j])
            medians[i] = medianY[0]

    ax.set_xticklabels(['Gull-Gull', 'Duck-Duck', 'Duck-Gull', 'Gull-Duck'])
    ax.set_title(plotTitle)
    ax.set_ylabel('Bit score' if distance == 'bit' else '% id')
    ax.set_ylim(0, 110)
    top = 105
    pos = np.arange(numBoxes)+1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    for tick, label in zip(range(numBoxes), ax.get_xticklabels()):
        ax.text(pos[tick], top, upperLabels[tick],
                horizontalalignment='center', size='small', weight='semibold',
                color='k')

    if stat:
        tggdd = stats.ttest_ind(gullgullDistances, duckduckDistances)
        tggdg = stats.ttest_ind(gullgullDistances, duckgullDistances)
        tgggd = stats.ttest_ind(gullgullDistances, gullduckDistances)
        tdddg = stats.ttest_ind(duckduckDistances, duckgullDistances)
        tddgd = stats.ttest_ind(duckduckDistances, gullduckDistances)
        tdggd = stats.ttest_ind(duckgullDistances, gullduckDistances)
        statistics = (('Unpaired t-test (p-values): \n gg-dd: %f \n gg-dg: %f '
                       '\n gg-gd: %f \n dd-dg: %f \n dd-gd: %f \n '
                       'dg-gd: %f \n') % (
            tggdd[1], tggdg[1], tgggd[1], tdddg[1], tddgd[1], tdggd[1]))
        plt.figtext(0.95, 0.1, statistics, color='black', size='medium')

    return (gullgullDistances, duckduckDistances,
            duckgullDistances, gullduckDistances)
