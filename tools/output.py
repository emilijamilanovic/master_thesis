import sys

# ****************************************************************************
# Output formats
# ****************************************************************************
class OutputFormat:
    def __init__(self):
        raise NotImplementedError

    def section(self, *args, **kwargs):
        raise NotImplementedError

    def image(self, *args, **kwargs):
        raise NotImplementedError

    def table(self, *args, **kwargs):
        raise NotImplementedError

    def chunk(self, *args, **kwargs):
        raise NotImplementedError

class OutputBuffer(OutputFormat):
    def __init__(self):
        self.content = []

    def section(self, name, depth):
        self.content.append(('sec', name, depth))

    def image(self, path):
        self.content.append(('fig', path))

    def table(self, name, path):
        self.content.append(('tab', name, path))

    def chunk(self, text):
        self.content.append(('chk', text))

    def write(self, fmt):
        for t, *a in self.content:
            if t == 'sec':
                fmt.section(*a)
            elif t == 'fig':
                fmt.image(*a)
            elif t == 'tab':
                fmt.table(*a)
            elif t == 'chk':
                fmt.chunk(*a)

class OutputMarkdown(OutputFormat):
    def __init__(self, path):
        if path == '-':
            self.f = sys.stdout
        else:
            self.f = open(path, 'w')

    def section(self, name, depth):
        self.f.write(f'\n\n{"#" * (depth + 1)} {name}\n\n')

    def image(self, path):
        self.f.write(f'\n![]({path})\n\n')

    def table(self, name, path):
        self.f.write(f'\n![{name}]({path})\n\n')

    def chunk(self, text):
        self.f.write(text)
        self.f.flush()

# ****************************************************************************
# Generic output
# ****************************************************************************
class Output:
    def __init__(self, formats = None):
        self.writers = formats if formats else []

    def section(self, name, depth = 0):
        for fmt in self.writers:
            fmt.section(name, depth)

    def image(self, path):
        for fmt in self.writers:
            fmt.image(path)

    def chunk(self, text):
        for fmt in self.writers:
            fmt.chunk(text)


# ****************************************************************************
#
# ****************************************************************************
