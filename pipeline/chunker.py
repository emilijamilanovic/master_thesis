#! /usr/bin/env python3
import sys
import re

class Chunk:
    def __init__(self, id=0, content=''):
        self.id = id
        self.content = content
    
    def annotated_string(self):
        return f'<{self.id}>\n{self.content}'

class Chunker:
    def __init__(self, chunk_title_min=100, chunk_min=200, chunk_max=2000):
        self.chunk_title_min = chunk_title_min
        self.chunk_min = chunk_min
        self.chunk_max = chunk_max

    @staticmethod
    def split_if(groups, fn):
        r = []
        for group in groups:
            if len(group) <= 1:
                r.append(group)
                continue
            
            l = [ group[0] ]
            prev = group[0]
            for obj in group[1:]:
                if fn(prev, obj):
                    r.append(l)
                    l = []
                l.append(obj)
                prev = obj
            
            r.append(l)
        return r
    
    @staticmethod
    def merge_if(groups, fn):
        r = []
        for group in groups:
            if len(group) <= 2:
                r.append(group)
                continue
            
            l = []
            prev = group[0]
            for obj in group[1:]:
                merged = fn(prev, obj)
                if merged is None:
                    l.append(prev)
                    prev = obj
                else:
                    prev = merged
            l.append(prev)
            r.append(l)
        return r

    def chunks(self, text):
        p = [ text.split('\n\n') ]
        
        p = self.split_if(p, lambda a, b: a[0:1] != '#' and b[0:1] == '#')
        
        p = self.merge_if(p, lambda a, b: a + '\n\n' + b if
            a[0:1] == '#' and
            len(a) < self.chunk_title_min and
            len(a) + len(b) <= self.chunk_max
            else None
        )
        
        p = self.merge_if(p, lambda a, b: a + '\n\n' + b if
            re.search(r'^[ \t]*:', b) and
            len(a) < self.chunk_min and
            len(a) + len(b) <= self.chunk_max
            else None
        )

        p = self.merge_if(p, lambda a, b: a + '\n\n' + b if
            re.search(r':\n*$', a) and
            len(a) + len(b) <= self.chunk_max
            else None
        )
        
        p = self.merge_if(p, lambda a, b: a + '\n\n' + b if
            len(a) < self.chunk_min and
            len(a) + len(b) <= self.chunk_max
            else None
        )
        
        p = self.merge_if(p, lambda a, b: a + '\n\n' + b if
            len(b) < self.chunk_min and
            len(a) + len(b) <= self.chunk_max
            else None
        )
        
        id = 0
        for group in p:
            for chunk in group:
                yield Chunk(id, chunk)
                id += 1

c = Chunker()

# NOTE: running this file directly chunks the RAW file text, which yields
# different chunk numbering than the pipeline (748 vs 754 chunks for pandoc.md).
# The pipeline loads sources through tools.source (SourceFile.fulltext()) first.
# Do NOT use this __main__ to regenerate pandoc.chunked.md — see the
# "Regenerating pandoc.chunked.md" recipe in the root README.
if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path, encoding='utf-8') as f:
            for chunk in c.chunks(f.read()):
                print(chunk.annotated_string(), end='\n\n')
