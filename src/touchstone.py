"""
A parser to parse the touchstone file format and store the data in a pandas DataFrame.
"""

from curses.ascii import isspace
import re
import pandas as pd

__all__ = ['TouchstoneParser']

optionsyntax = re.compile(r'(?<=\s|#)(?:([kMG]?Hz)|([SYZGH])|(DB|MA|RI)|(?:R\s+([\d\.]+)))(?=\s|$)', re.I | re.A)


class TouchstoneParser():
    def __init__(self, port_num):
        """
        Initialize the parser with the given number of ports.

        Args:
        - port_num: The number of ports in the touchstone file.
        """
        self.port_num = port_num
        self.options = {"freq_unit": "Hz", "mode": "s", "format": "db", "r": 50}
        self.columns = self.init_columns()
    
    def init_columns(self):
        """
        Initialize the columns of the pandas DataFrame.
        """
        columns = ['freq']
        for i in range(self.port_num):
            for j in range(self.port_num):
                columns.append(f'{i+1}{j+1}_1')
                columns.append(f'{i+1}{j+1}_2')
        return columns
        
    def parse_option(self, line):
        """
        Parse the option line of the touchstone file.
        
        Args:
        - line: The option line to be parsed.
        """
        for match in optionsyntax.finditer(line):
            if match.group(1):
                self.options['freq_unit'] = match.group(1)
            elif match.group(2):
                self.options['mode'] = match.group(2).lower()
            elif match.group(3):
                self.options['format'] = match.group(3).lower()
            elif match.group(4):
                self.options['r'] = float(match.group(4))
        return 
    
    def parse_data(self, line):
        """
        Parse the data line of the touchstone file.

        Args:
        - line: The data line to be parsed.
        """
        splits = line.split()
        freq = float(splits[0])

        # change the unit of frequency
        if self.options['freq_unit'].lower() == 'khz':
            freq *= 1e3
        elif self.options['freq_unit'].lower() == 'mhz':
            freq *= 1e6
        elif self.options['freq_unit'].lower() == 'ghz':
            freq *= 1e9

        row = []
        for i in range(1, len(splits), 2):
            row.extend([float(splits[i]), float(splits[i+1])])
        return [freq] + row

    def parse(self, filename):
        """
        Parse the given touchstone file and return the data in a dictionary.
        
        Args:
        - filename: The name of the touchstone file to be parsed.
        
        Returns:
        - A dictionary containing the parsed data.
        """
        data = pd.DataFrame(columns=self.columns)
        i = 0
        with open(filename, 'r') as f:
            lines = f.readlines()
            for _, line in enumerate(lines):
                if line.startswith('!'):
                    # comment line
                    continue
                elif line.startswith('#'):
                    # parse the first option line
                    self.parse_option(line)
                elif line.isspace():
                    # empty line
                    continue
                else:
                    row = self.parse_data(line)
                    # append a row to the data DataFrame
                    data.loc[i] = row
                    i += 1
        return data