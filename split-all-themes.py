"""Reproducible within-theme holdout, keeping upper/middle pairs together."""
from collections import defaultdict
from pathlib import Path
import json
import random

path = Path(__file__).with_name('corpus-common.json')
data = json.loads(path.read_text())
records = data['corpusRecords']
seed = 731
rng = random.Random(seed)
by_scene = defaultdict(lambda: defaultdict(list))
for index, record in enumerate(records):
    pair = tuple(record['text'].split('／')[:2])
    by_scene[record['scene']][pair].append(index)

# Keep the existing two training and two evaluation display examples.
reserved = {}
for text, split in zip(data['texts'], ['development', 'development', 'test', 'test']):
    record = next(record for record in records if record['text'] == text)
    reserved[(record['scene'], tuple(text.split('／')[:2]))] = split

evaluation = set()
for scene, pairs in by_scene.items():
    selected = [indices for pair, indices in pairs.items()
                if reserved.get((scene, pair)) == 'test']
    remaining = 51 - sum(map(len, selected))
    candidates = [(pair, indices) for pair, indices in pairs.items()
                  if (scene, pair) not in reserved]
    rng.shuffle(candidates)
    # Subset sum selects whole pair groups while retaining exactly 51 works.
    choices = {0: []}
    for pair, indices in candidates:
        for count, groups in list(choices.items()):
            new_count = count + len(indices)
            if new_count <= remaining and new_count not in choices:
                choices[new_count] = groups + [indices]
        if remaining in choices:
            break
    assert remaining in choices, f'Cannot allocate 51 evaluation works for {scene}'
    selected += choices[remaining]
    evaluation.update(index for indices in selected for index in indices)

data['testIndices'] = sorted(evaluation)
data['trainIndices'] = [i for i in range(len(records)) if i not in evaluation]
for index, record in enumerate(records):
    record['split'] = 'test' if index in evaluation else 'development'
data['exampleSplits'] = [next(record['split'] for record in records
                              if record['text'] == text) for text in data['texts']]
groups = list(dict.fromkeys(record['group'] for record in records))
meta = data['corpusMeta']
meta.update(testGroups=groups, developmentGroups=groups,
            testCount=len(evaluation), trainCount=len(data['trainIndices']),
            splitSeed=seed, splitStrategy='within-scene-upper-middle-grouped',
            splitMethod='14場面すべてを学習・評価の両方に含める。各場面51件を評価用に保留。'
                        '同じ場面で上五と中七が一致する作品群は分割しない。'
                        '候補句自体は両側で共有するため、既知テーマ内の新しい組み合わせの評価。')

assert len(data['trainIndices']) == 4286 and len(evaluation) == 714
assert len({record['text'] for record in records}) == 5000
training_chars = set(''.join(records[i]['text'] for i in data['trainIndices']))
evaluation_chars = set(''.join(records[i]['text'] for i in evaluation))
assert not evaluation_chars - training_chars
for scene, pairs in by_scene.items():
    assert sum(index in evaluation for indices in pairs.values() for index in indices) == 51
    assert all(len({records[i]['split'] for i in indices}) == 1 for indices in pairs.values())
path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
print('14 themes in both splits; train 4,286 / evaluation 714; no evaluation-only characters.')
