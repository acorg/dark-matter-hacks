from unittest import TestCase
from json import dumps
from mock import patch

from scripts import nicola
from mocking import mockOpen


class TestGetBird(TestCase):
    """
    Tests for the _getBird function.
    """
    def test_getBirdAllNoBird(self):
        title = "I'm a random title and don't contain anything"
        result = nicola._getBird(title, colorBy='all')
        self.assertEqual(result, nicola.NEITHER)

    def test_getBirdAnsCharNoBird(self):
        title = "I'm a random title and don't contain anything"
        result = nicola._getBird(title, colorBy='anseriformes '
                                                'and charadriiformes')
        self.assertEqual(result, nicola.OTHER)

    def test_getBirdAnsChar(self):
        title = "I'm an anatinae and I'm a teal"
        result = nicola._getBird(title, colorBy='anseriformes '
                                                'and charadriiformes')
        self.assertEqual(result, nicola.ANATINAE)

    def test_getBirdAll(self):
        title = "I'm from all and I'm a tern"
        result = nicola._getBird(title, colorBy='all')
        self.assertEqual(result, nicola.CHARADRIIFORMES)


class TestColorByContinent(TestCase):
    """
    Tests for the _getCountry function.
    """
    def test_getCountryWithCountry(self):
        title = "I'm/a randomtitle I'm from/Mongolia"
        result = nicola._getCountry(title)
        self.assertEqual(result, nicola.EURASIA)

    def test_getCountryWithWrongCountry(self):
        title = "I'm/a randomtitle I'm from/space"
        result = nicola._getCountry(title)
        self.assertEqual(result, 'white')


class TestComputePercentIdIdentity(TestCase):
    """
    Tests for the computePercentId function.
    """
    def testComputePercentIdIdentical(self):
        params = {
            'application': 'BLASTN',
        }
        record = {
            "query": "H6E8I1T01BFUH9",
            "alignments": [
                {
                    "length": 2885,
                    "hsps": [
                        {
                            'bits': 20,
                            'sbjct_end': 15400,
                            'expect': 3.29804,
                            'sbjct': 'TACCCTGCGGCCCGCTACGGCTGG',
                            'sbjct_start': 15362,
                            'query': 'TACCCTGCGGCCCGCTACGGCTGG',
                            'frame': [1, 1],
                            'query_end': 68,
                            'query_start': 28
                        }
                    ],
                    "title": "Merkel1"
                }
            ]
        }

        mockOpener = mockOpen(read_data=dumps(params) + '\n' +
                              dumps(record) + '\n')
        with patch('__builtin__.open', mockOpener, create=True):
            records = list(nicola._records('file.json'))
            result = nicola.computePercentId(records[0].alignments[0])
            self.assertEqual(result, 100.0)

    def testComputePercentIdDifferent(self):
        params = {
            'application': 'BLASTN',
        }
        record = {
            "query": "H6E8I1T01BFUH9",
            "alignments": [
                {
                    "length": 2885,
                    "hsps": [
                        {
                            'bits': 20,
                            'sbjct_end': 15400,
                            'expect': 3.29804,
                            'sbjct': '------------------------',
                            'sbjct_start': 15362,
                            'query': 'TACCCTGCGGCCCGCTACGGCTGG',
                            'frame': [1, 1],
                            'query_end': 68,
                            'query_start': 28
                        }
                    ],
                    "title": "Merkel1"
                }
            ]
        }

        mockOpener = mockOpen(read_data=dumps(params) + '\n' +
                              dumps(record) + '\n')
        with patch('__builtin__.open', mockOpener, create=True):
            records = list(nicola._records('file.json'))
            result = nicola.computePercentId(records[0].alignments[0])
            self.assertEqual(result, 0.0)

    def testComputePercentIdRealistic(self):
        params = {
            'application': 'BLASTN',
        }
        record = {
            "query": "H6E8I1T01BFUH9",
            "alignments": [
                {
                    "length": 2885,
                    "hsps": [
                        {
                            'bits': 20,
                            'sbjct_end': 15400,
                            'expect': 3.29804,
                            'sbjct': 'TACCCTGCGGCCCGC-ACGGCTGG',
                            'sbjct_start': 15362,
                            'query': 'TACCCTGCGGCCCGCTACGGCTGG',
                            'frame': [1, 1],
                            'query_end': 68,
                            'query_start': 28
                        }
                    ],
                    "title": "Merkel1"
                }
            ]
        }

        mockOpener = mockOpen(read_data=dumps(params) + '\n' +
                              dumps(record) + '\n')
        with patch('__builtin__.open', mockOpener, create=True):
            records = list(nicola._records('file.json'))
            result = nicola.computePercentId(records[0].alignments[0])
            self.assertEqual(result, 95.83333333333334)


class TestDistancePlot(TestCase):
    """
    Testing whether the distancePlot function returns the right thing
    """
    def testDistancePlot(self):
        params = {
            'application': 'BLASTN',
        }
        record = {
            "query": "H6E8I1T01BFUH9",
            "alignments": [
                {
                    "length": 2885,
                    "hsps": [
                        {
                            'bits': 20,
                            'sbjct_end': 15400,
                            'expect': 3.29804,
                            'sbjct': 'TACCCTGCGGCCCGC-ACGGCTGG',
                            'sbjct_start': 15362,
                            'query': 'TACCCTGCGGCCCGCTACGGCTGG',
                            'frame': [1, 1],
                            'query_end': 68,
                            'query_start': 28
                        }
                    ],
                    "title": "gb:A2|Orm:Ins A/Africling/England-Q/"
                             "983/1979|Set:7|Sue:H7N1|Host:Africling"
                },
                {
                    "length": 2885,
                    "hsps": [
                        {
                            'bits': 20,
                            'sbjct_end': 15400,
                            'expect': 3.29804,
                            'sbjct': '------------------------',
                            'sbjct_start': 15362,
                            'query': 'TACCCTGCGGCCCGCTACGGCTGG',
                            'frame': [1, 1],
                            'query_end': 68,
                            'query_start': 28
                        }
                    ],
                    "title": "gb:A2|Orm:Ins A/Africling/England-Q/"
                             "983/1979|Set:7|Sue:H7N2|Host:Africling"
                }
            ]
        }
        mockOpener = mockOpen(read_data=dumps(params) + '\n' +
                              dumps(record) + '\n')
        with patch('__builtin__.open', mockOpener, create=True):
            records = list(nicola._records('file.json'))
            for record in records:
                dist, distances = nicola.distancePlot(record)
                self.assertEqual(dist, 20)
                self.assertEqual(2, distances)
