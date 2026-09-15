import os
import re

# ****************************************************************************
# SourceFile
# ****************************************************************************
class SourceFile:
    def __init__(self, path, directory):
        self.path = path
        self.directory = directory
        self.par = [ '' ]

        self.load(path)

    LINKS = re.compile(r'!\[([^]|]*)\]\(([^)|]+)\)')

    def append(self, line, makepar = False):
        line = line.rstrip()

        if makepar or line == '':
            if self.par[-1] != '':
                self.par.append('')

        if line != '':
            self.par[-1] += ('' if self.par[-1] == '' else '\n') + line

            if makepar:
                self.par.append('')

    def load_line(self, line):
        self.append(line)
        return

        pos = 0

        for m in re.finditer(self.LINKS, line):
            start, stop = m.span(0)
            text = m.group(1)
            url = m.group(2)

            self.append(line[pos:start])

            cap = 'TABLE ' if url.endswith('.csv') else 'IMAGE '

            self.append(cap + url, makepar = True)

            pos = stop

        self.append(line[pos:])

    def load(self, path):
        self.load_line('')

        with open(path) as f:
            for line in f:
                self.load_line(line)

    def fulltext(self):
        return '\n\n'.join(self.par)

    def chunks(self, length=5000, overlap=1000):
        window = []

        for par in self.par:
            window.append(par)

            if sum(len(p) for p in window) > length:
                r = '\n\n'.join(window) + '\n\n'

                while len(window) > 1 and sum(len(p) for p in window) > overlap:
                    window = window[1:]

                yield r

        yield '\n\n'.join(window) + '\n\n'

# ****************************************************************************
# Source
# ****************************************************************************
class Source:
    def __init__(self, directory=None):
        self.directories = []
        self.cache = {}
        self.files = []

        if directory is not None:
            self.directories.append(directory)
            self.scan(self.directories)


    def add(self, filename=None, directory=None, path=None):
        if filename is None and path is None:
            raise ValueError('Either filename or path must be set')

        if filename is None:
            filename = os.path.basename(path)

        if path is None:
            if directory is None:
                path = filename
            else:
                path = os.path.join(directory, f)

        if directory is None:
            directory = os.path.dirname(path)

        self.files.append((filename, directory, path))

    def scan(self, dirlist):
        for directory in dirlist:
            flist = os.listdir(directory)
            for f in flist:
                if f.endswith('_pan.md') or not f.endswith('.md'):
                    continue

                path = os.path.join(directory, f)

                if not os.path.isfile(path):
                    continue

                self.add(f, directory, path)

        #print('Sources: ' + ', '.join(self.paths)
        #    + ' -> ' + ', '.join(s[0] for s in self.files))

    def match(self, name):
        return [ (f, d, p) for f, d, p in self.files if f.startswith(name) ]

    def count(self, name):
        return len(self.match(name))

    def get(self, name, index=0):
        matches = self.match(name)

        if index >= len(matches):
            raise ValueError(f'\n\nNo match found for "{name}" '
                + f'in directories {self.directories}\n\n'
                + 'Check that the directory contains this source\n')

        f, d, p = self.match(name)[index]

        if p not in self.cache:
            self.cache[p] = SourceFile(p, d)

        return self.cache[p]
