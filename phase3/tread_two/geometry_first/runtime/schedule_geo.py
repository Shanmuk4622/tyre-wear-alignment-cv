"""Full 386-source coverage, six marked padding draws, deterministic component mixing."""
from collections import Counter
import numpy as np
from common import digest


def schedule(rows, seed=1, epoch=1):
    rows = sorted(rows, key=lambda r: r['image_id'])
    by_id = {r['image_id']: r for r in rows}
    assert len(rows) == len(by_id) == 386
    groups = Counter(r['component'] for r in rows)
    largest = sorted(groups, key=lambda x: (-groups[x], x))[0]
    rng = np.random.default_rng(seed * 1000003 + epoch)
    a = list(rng.permutation([r['image_id'] for r in rows if r['component'] == largest]))
    c = list(rng.permutation([r['image_id'] for r in rows if r['component'] != largest]))
    assert (len(a), len(c)) == (249, 137)
    batches = [[(a[i], False), (c[i], False)] for i in range(49)]
    rest = [(x, False) for x in a[49:] + c[49:]]
    rest += [(x, True) for x in rng.permutation(sorted(by_id))[:6]]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    for bi, batch in enumerate(batches):
        while len(batch) < 8:
            present = {x for x, _ in batch}
            found = next((j for j, v in enumerate(rest) if v[0] not in present), None)
            if found is not None:
                batch.append(rest.pop(found))
                continue
            repaired = False
            for prior in batches[:bi]:
                for slot in range(2, len(prior)):
                    if prior[slot][0] in present:
                        continue
                    replacement = next((j for j, v in enumerate(rest) if v[0] not in {x for x, _ in prior}), None)
                    if replacement is not None:
                        candidate = prior[slot]
                        prior[slot] = rest.pop(replacement)
                        batch.append(candidate)
                        repaired = True
                        break
                if repaired:
                    break
            assert repaired, 'Cannot pack a duplicate-free batch'
    assert not rest
    assert Counter(x for batch in batches for x, pad in batch if not pad) == Counter(by_id.keys())
    assert sum(pad for batch in batches for _, pad in batch) == 6
    output = []
    for bi, batch in enumerate(batches):
        assert len({x for x, _ in batch}) == 8 and len({by_id[x]['component'] for x, _ in batch}) >= 2
        output.append([{'image_id': str(x), 'component': by_id[x]['component'], 'padding': bool(pad),
                        'epoch': epoch, 'batch': bi, 'slot': slot,
                        'seed': int(seed * 10000019 + epoch * 100003 + bi * 8 + slot)}
                       for slot, (x, pad) in enumerate(batch)])
    return output, digest(output)
