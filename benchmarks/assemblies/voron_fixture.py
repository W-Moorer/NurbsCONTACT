"""Extract the original Front Idler B subtree from the pinned Voron STEP."""

import argparse
import hashlib
import mmap
import re
import shutil
from array import array
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


REVISION = "7fcb3257a3063225934d9608f82037aa6587d844"
URL = f"https://raw.githubusercontent.com/VoronDesign/Voron-Trident/{REVISION}/CAD/trident_assembly.step.zip"
ZIP_SHA256 = "9fe5a74f77048603e9289bdfcfa0db60b443252b9236ac3ef39821dc2a91209a"
STEP_SHA256 = "416eda4e9212b894ed6f78ad3a6af55757115ebfcc4c18db933743d97dd48ccf"
SUBASSEMBLY_SHA256 = "c1abfdd0d77eaaa2ac4e90cdfd798171275001b650660f7dcf2a7222300f93a7"
SUBASSEMBLY_ID = 2547975
ID_LIMIT = 3_000_000
ENTITY = re.compile(rb"^#([0-9]+)=[ \t]*", re.MULTILINE)
REFERENCE = re.compile(rb"'(?:''|[^'])*'|#([0-9]+)")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_step(root):
    step = root/"voron-trident.step"
    if not step.exists() or sha256(step) != STEP_SHA256:
        archive = root/"voron-trident.step.zip"
        if not archive.exists() or sha256(archive) != ZIP_SHA256:
            with urlopen(URL, timeout=100) as response, archive.open("wb") as target:
                shutil.copyfileobj(response, target)
        if sha256(archive) != ZIP_SHA256:
            raise ValueError("Voron source archive hash differs")
        with ZipFile(archive) as bundle, bundle.open("trident_assembly.step") as source, step.open("wb") as target:
            shutil.copyfileobj(source, target)
    if sha256(step) != STEP_SHA256:
        raise ValueError("Voron source STEP hash differs")
    return step


def references(statement):
    return [int(item[1]) for item in REFERENCE.finditer(statement) if item[1]]


def extract(source, target):
    parts = defaultdict(list)
    product_shapes, placements, representations, shape_links = (
        defaultdict(list) for _ in range(4)
    )
    starts, ends = (array("I", [0])*ID_LIMIT for _ in range(2))
    with source.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header = data.find(b"\nDATA;")
        data_end = data.rfind(b"ENDSEC;")
        assert 0 < header < data_end
        header_end = header+len(b"\nDATA;")
        for match in ENTITY.finditer(data, header_end, data_end):
            idx = int(match[1])
            starts[idx] = match.start()
            # Each entity terminates before the next entity or ENDSEC.
            # Find the first semicolon outside a quoted label if needed.
            end = data.find(b";", match.end(), data_end)
            assert end != -1
            if b"'" in data[match.end():end] and data[match.end():end].count(b"'") % 2:
                while data[match.end():end].count(b"'") % 2:
                    end = data.find(b";", end+1, data_end)
                    assert end != -1
            ends[idx] = end+1
            tag = data[match.end():match.end()+42].split(b"(", 1)[0]
            if tag == b"NEXT_ASSEMBLY_USAGE_OCCURRENCE":
                parent, child = references(data[match.end():end])
                parts[parent].append((idx, child))
            elif tag == b"PRODUCT_DEFINITION_SHAPE":
                definition, = references(data[match.end():end])
                product_shapes[definition].append(idx)
            elif tag == b"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION":
                _, occurrence_shape = references(data[match.end():end])
                placements[occurrence_shape].append(idx)
            elif tag == b"SHAPE_DEFINITION_REPRESENTATION":
                definition, representation = references(data[match.end():end])
                representations[definition].append((idx, representation))
            elif tag == b"SHAPE_REPRESENTATION_RELATIONSHIP":
                a, b = references(data[match.end():end])
                shape_links[a].append((idx, b))
                shape_links[b].append((idx, a))

        assert b"'Front Idler B'" in data[starts[SUBASSEMBLY_ID]:ends[SUBASSEMBLY_ID]]
        products, occurrences = set(), set()
        pending = [SUBASSEMBLY_ID]
        while pending:
            product = pending.pop()
            if product in products:
                continue
            products.add(product)
            for child_occurrence, child_product in parts[product]:
                occurrences.add(child_occurrence)
                pending.append(child_product)

        needed = products | occurrences
        rep_tasks = []
        for product in products:
            assert product_shapes[product]
            for pds in product_shapes[product]:
                needed.add(pds)
                for link, rep in representations[pds]:
                    needed.add(link)
                    rep_tasks.append(rep)
        for occurrence in occurrences:
            assert len(product_shapes[occurrence]) == 1
            pds = product_shapes[occurrence][0]
            assert len(placements[pds]) == 1
            needed.update((pds, placements[pds][0]))
        selected_reps = set()
        while rep_tasks:
            rep = rep_tasks.pop()
            if rep in selected_reps:
                continue
            selected_reps.add(rep)
            for link, relation in shape_links[rep]:
                needed.add(link)
                rep_tasks.append(relation)

        pending = list(needed | selected_reps)
        closure = set()
        while pending:
            idx = pending.pop()
            if idx in closure:
                continue
            assert starts[idx] and ends[idx], idx
            closure.add(idx)
            statement = data[starts[idx]:ends[idx]]
            pending.extend(ref for ref in references(statement.split(b"=", 1)[1]) if ref not in closure)
        with target.open("wb") as out:
            out.write(data[:header_end]+b"\n")
            for idx in sorted(closure, key=starts.__getitem__):
                out.write(data[starts[idx]:ends[idx]]+b"\n")
            out.write(b"ENDSEC;\nEND-ISO-10303-21;\n")
    result = sha256(target)
    if result != SUBASSEMBLY_SHA256:
        raise ValueError("extracted Voron STEP hash differs")
    print(f"{target}: {len(occurrences)} occurrence links, {len(closure)} original entities, sha256 {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    extract(source_step(args.work), args.work/"voron-front-idler-b.step")


if __name__ == "__main__":
    main()
