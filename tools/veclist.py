#! /usr/bin/env python3
import sys
import os
import struct
import argparse
import re
import json
import numpy as np
#import gzip

# ****************************************************************************
# Cosine similarity
# ****************************************************************************
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# ****************************************************************************
# Document vector
# ****************************************************************************
class DocVector:
    def __init__(self, doc_id, vector, metadata=None):
        self.doc_id = doc_id
        self.vector = vector
        self.metadata = metadata

    def show(self):
        s_doc_id = '"' + self.doc_id + '"'
        s_vector = ' '.join(f'{v:6.3f}' for v in self.vector[:2])
        s_metadata = str(self.metadata)

        if len(s_metadata) > 32:
            s_metadata = s_metadata[:29] + '...'

        print(f'{s_doc_id:>27} [{s_vector} ...] {s_metadata:32}')

# ****************************************************************************
# Vector list
# ****************************************************************************
class VectorList:
    def __init__(self):
        self.docv = []

    def add(self, doc_id, vector, metadata=None):
        self.docv.append(DocVector(doc_id, vector, metadata))

    def count(self):
        return len(self.docv)

    def serialize(self):
        r = bytearray()

        for v in self.docv:
            doc_id = v.doc_id.encode()
            metadata = json.dumps(v.metadata).encode()

            r += struct.pack('<QQQ', len(doc_id), len(metadata), len(v.vector))
            r += doc_id
            r += metadata

            for f in v.vector:
                r += struct.pack('<f', f)

        return r

    def deserialize(self, data):
        class DataReader:
            def __init__(self, data):
                self.data = data
                self.offset = 0

            def get(self, size):
                r = data[self.offset:self.offset+size]
                self.offset += size
                return r

            def left(self):
                return len(self.data) - self.offset

        reader = DataReader(data)

        while reader.left() > 0:
            doc_s, meta_s, n = struct.unpack('<QQQ', reader.get(24))

            doc_id = reader.get(doc_s).decode()
            metadata = json.loads(reader.get(meta_s).decode())
            vector = [ 0.0 ] * n

            for j in range(n):
                vector[j], = struct.unpack('<f', reader.get(4))

            self.add(doc_id, vector, metadata)

    def dim(self):
        if len(self.docv) == 0:
            return 0
        return len(self.docv[0].vector)

    def stats(self):
        print(f'    Vectors: {len(self.docv):12}')
        print(f'        Dim: {self.dim():12}')

    def show(self):
        self.stats()

        for v in self.docv:
            v.show()

    def load(self, path):
        if path == '-':
            data = sys.stdin.buffer.read()
        else:
            with open(path, 'rb') as f:
                data = f.read()

        #data = gzip.decompress(data)
        self.deserialize(data)

    def save(self, path):
        data = self.serialize()
        #data = gzip.compress(data)

        if path == '-':
            sys.stdout.buffer.write(data)
        else:
            with open(path, 'wb') as f:
                f.write(data)

    def doc_ids(self):
        r = set()

        for v in self.docv:
            r.add(v.doc_id)

        return sorted(r)

    def filter(self, fn=None, doc_id=None, metadata=None):
        r = VectorList()

        for v in self.docv:
            if fn and not fn(v):
                continue

            if doc_id and v.doc_id != doc_id:
                continue

            if metadata:
                take = True

                for key, value in metadata.items():
                    if key not in v.metadata:
                        take = False
                        break
                    if v.metadata[key] != value:
                        take = False
                        break

                if not take:
                    continue

            r.add(v.doc_id, v.vector, v.metadata)

        return r

    def selection_show(self, sim, size, take, total):
        totalall = sum(size)
        print(f'============== total {total:9} / {totalall:9} ==============')

        for i in range(len(take)):
            sep = '+++' if take[i] else '---'
            c0 = '\x1b[92m' if take[i] else '\x1b[31m'

            print(f'\x1b[90m{sep} {sim[i]:6.3f} | {size[i]:6} {sep}\x1b[0m')

            par = self.docv[i].metadata['chunk'].replace('\n', '\n' + c0)

            print(f'{c0}{par}\x1b[0m')

    def selection(self, vector, context, target_max):
        if type(context) is int:
            ctx_before = context
            ctx_after = context
        else:
            ctx_before = context[0]
            ctx_after = context[1]

        size = [ len(v.metadata['chunk']) for v in self.docv ]
        sim = [ cosine_similarity(vector, v.vector) for v in self.docv ]
        prio = sorted(range(len(sim)), key=lambda k: -sim[k])
        take = [ False for i in range(len(self.docv)) ]

        total = 0

        for i in prio:
            add = 0

            for j in range(i - ctx_before, i + ctx_after):
                if j < 0 or j >= len(take):
                    continue

                if not take[j]:
                    add += size[j]

            if total + add > target_max:
                break

            for j in range(i - ctx_before, i + ctx_after):
                if j < 0 or j >= len(take):
                    continue
                take[j] = True

            total += add

        self.selection_show(sim, size, take, total)

        return take

    def fulltext(self):
        return '\n'.join(v.metadata['chunk'] for v in self.docv)


    def select(self, *args, **kwargs):
        take = self.selection(*args, **kwargs)

        return '\n'.join(
            self.docv[i].metadata['chunk']
            for i in range(len(self.docv))
            if take[i]
        )


# ****************************************************************************
# Test vectors
# ****************************************************************************
def test(vecs):
    vecs.add('doc_a', [ 0.0, 0.0, 1.0 ])
    vecs.add('doc_b', [ 0.0, 1.0, 0.0 ], { 'page': 1 })
    vecs.add('doc_b', [ 0.0, 1.0, 1.0 ], { 'page': 2 })
    vecs.add('doc_b', [ 1.0, 0.0, 0.0 ], { 'page': 3 })
    vecs.add('doc_c', [ 1.0, 0.0, 1.0 ])

# ****************************************************************************
# Main
# ****************************************************************************
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', help='stats|show|merge')
    ap.add_argument('file', help='Input vector list file', nargs='*')
    ap.add_argument('-o', '--output', help='Output vector list file')
    args = ap.parse_args()

    vecs = VectorList()

    for path in args.file:
        vecs.load(path)

    if args.action == 'test':
        test(vecs)
    elif args.action == 'stats':
        vecs.stats()
        args.output = None
    elif args.action == 'show':
        vecs.show()
        args.output = None
    elif args.action == 'merge':
        pass
    else:
        print(f'Unknown action: "{args.action}"')
        exit(1)

    if args.output is not None:
        vecs.save(args.output)

if __name__ == '__main__':
    main()

