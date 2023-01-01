import unittest
from incidence_graph import IncidenceGraph


class MyTestCase(unittest.TestCase):
    def test_empty_creation(self):
        g = IncidenceGraph()
        self.assertEqual(g.size(0), 0, 'Empty graph should have no vertices')
        self.assertEqual(len(g), 0, 'Empty graph should have no nodes')
        self.assertSequenceEqual(g.shape(), [0], 'Empty graph should have shape [0]')


if __name__ == '__main__':
    unittest.main()
