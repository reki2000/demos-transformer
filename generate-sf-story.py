"""Generate the SF short-story corpus (corpus-sf.json) in the corpus-common.json format.

Each story is four sentences, 起承転結:
  起 introduces the setting, the hero and a keepsake.
  承 brings a problem (lost signal, failing oxygen, lost way).
  転 turns it: the keepsake solves it, a hidden cause is revealed,
     or a stranger appears.
  結 resolves that problem in the way the turn set up.

Later sentences repeat the hero's name, the keepsake and setting facts
(signal delay, what covers the antenna, what lights the way), so
check_story() can tell whether a story, for example one the model wrote,
holds together.
"""
from pathlib import Path
import json
import random

SEED = 2026
SPLIT_SEED = 731
COUNT = 5000
MAX_CONTEXT = 128  # tokens fed to the model: 〈始〉 + text (targets: text + 〈終〉)

SETTINGS = {
    'mars': dict(scene='火星', place='火星の基地', delay='二十分後',
                 dust='赤い砂', glow='砂嵐の向こうの青い光'),
    'moon': dict(scene='月面', place='月面の観測所', delay='一秒後',
                 dust='細かい灰色の砂', glow='地球の照り返し'),
    'europa': dict(scene='エウロパ', place='エウロパの氷上基地', delay='一時間後',
                   dust='厚い霜', glow='氷の下から差す青い光'),
    'titan': dict(scene='タイタン', place='タイタンの湖畔基地', delay='一時間半後',
                  dust='黒いもや', glow='霧の中を漂う光の粒'),
    'station': dict(scene='軌道ステーション', place='軌道ステーション', delay='数秒後',
                    dust='宇宙のちり', glow=None),
}

NAMES = ['ミナ', 'ケン', 'ソラ', 'ユウ', 'リコ', 'ハル', 'ノア', 'アキ']

# key: (introduced as, referred to later as)
ITEMS = {
    'radio': ('祖母のラジオ', 'ラジオ'),
    'seed': ('一粒の種', '種'),
    'watch': ('父の時計', '時計'),
    'photo': ('古い写真', '写真'),
    'letter': ('母の手紙', '手紙'),
}

KI = ['{place}で、{n}は{item}をいつも持ち歩いていた。',
      '{n}は{place}でひとり働き、{item}だけを友にしていた。']

SHO = {
    'signal': ['ある日、地球との通信が突然途絶えた。',
               '嵐のあと、地球からの声が聞こえなくなった。'],
    'oxygen': ['ある夜、酸素装置が壊れ、残りは三日分になった。',
               '警報が鳴り、酸素が漏れていると表示された。'],
    'lost': ['外の調査中、{n}は基地への道を見失った。',
             '探査車が止まり、{n}は基地から遠く取り残された。'],
}
# Nobody walks home outside an orbital station.
EXCLUDED = {('station', 'lost')}

# The keepsake turns the story only for these problems.
SOLVES = {
    ('signal', 'radio'): ('{n}はラジオの部品で送信機を組み直した。',
                          '{delay}、地球から「聞こえるよ」と返事が来た。'),
    ('signal', 'letter'): ('手紙の隅に、古い非常用の周波数が書かれていた。',
                           'その周波数で呼ぶと、{delay}に母の声が返ってきた。'),
    ('oxygen', 'seed'): ('{n}は種を水耕槽にまき、葉に酸素を作らせた。',
                         '{n}は緑の葉に囲まれて、救助の日を待った。'),
    ('oxygen', 'radio'): ('{n}はラジオで近くを通る貨物船を呼んだ。',
                          '三日後、貨物船が{n}を迎えに来た。'),
    ('lost', 'watch'): ('時計の針と太陽の位置から、{n}は方角を割り出した。',
                        '{n}は時計を握りしめ、夜明け前に基地へ戻った。'),
    ('lost', 'photo'): ('写真に写る山の形が、遠くの稜線と重なった。',
                        '{n}は写真の山を目指して歩き、基地にたどり着いた。'),
}

# twist -> problem -> (転, 結)
TWISTS = {
    'reveal': {
        'signal': ('原因は、アンテナに積もった{dust}だった。',
                   '{n}が{dust}を払うと、{delay}に地球の声が戻った。'),
        'oxygen': ('だが本当に壊れていたのは、酸素計の方だった。',
                   '酸素はまだ百日分あり、{n}は{ref}を見て笑った。'),
        'lost': ('足元の足跡は、さっき通った{n}自身のものだった。',
                 '{n}は足跡を逆にたどり、{ref}を握って基地に戻った。'),
    },
    'stranger': {
        'signal': ('そのとき、知らない言葉の信号が届いた。',
                   '{n}は{ref}をそばに置き、初めての返事を送った。'),
        'oxygen': ('そのとき、扉の外に見知らぬ影が立っていた。',
                   '影が置いた酸素の箱で、{n}は救助まで生き延びた。'),
        'lost': ('{glow}が、{n}の前に道を照らした。',
                 '光を追って基地に戻り、{n}は{ref}にそっと礼を言った。'),
    },
}


def twists_for(problem, item):
    return (['item'] if (problem, item) in SOLVES else []) + list(TWISTS)


def compose(setting, name, item, problem, twist, choice):
    """Return [(sentence, facts it implies)] for one choice of every slot."""
    st = SETTINGS[setting]
    ki, sho = choice
    fill = dict(n=name, item=ITEMS[item][0], ref=ITEMS[item][1], **st)
    ten, ketsu = SOLVES[problem, item] if twist == 'item' else TWISTS[twist][problem]

    def facts(template, **base):
        # A sentence pins down the slots its placeholders name.
        if '{n}' in template:
            base['name'] = name
        if '{ref}' in template or twist == 'item':
            base['item'] = item
        if any(k in template for k in ('{delay}', '{dust}', '{glow}')):
            base['setting'] = setting
        return base

    return [
        (KI[ki].format(**fill), dict(setting=setting, name=name, item=item)),
        (SHO[problem][sho].format(**fill), facts(SHO[problem][sho], problem=problem)),
        (ten.format(**fill), facts(ten, problem=problem, twist=twist)),
        (ketsu.format(**fill), facts(ketsu, problem=problem, twist=twist)),
    ]


def all_slots():
    for setting in SETTINGS:
        for problem in SHO:
            if (setting, problem) in EXCLUDED:
                continue
            for item in ITEMS:
                for twist in twists_for(problem, item):
                    for name in NAMES:
                        for choice in [(k, s) for k in range(len(KI))
                                       for s in range(len(SHO[problem]))]:
                            yield dict(setting=setting, name=name, item=item,
                                       problem=problem, twist=twist, choice=choice)


def split_sentences(text):
    parts = text.split('。')
    return [p + '。' for p in parts[:-1]] if parts[-1] == '' else None


_TABLE = None


def check_story(text):
    """Classify a story: 'consistent', 'inconsistent' or 'unknown-sentence'.

    A sentence the generator never produces makes it 'unknown-sentence'.
    Otherwise the story is consistent when one assignment of setting,
    hero, keepsake, problem and turn explains all four sentences.
    """
    global _TABLE
    if _TABLE is None:
        _TABLE = [dict() for _ in range(4)]
        for slots in all_slots():
            for pos, (sentence, facts) in enumerate(compose(**slots)):
                _TABLE[pos].setdefault(sentence, set()).add(tuple(sorted(facts.items())))
    sentences = split_sentences(text)
    if not sentences or len(sentences) != 4 or \
            any(s not in _TABLE[i] for i, s in enumerate(sentences)):
        return 'unknown-sentence'
    states = [{}]
    for i, sentence in enumerate(sentences):
        states = [{**state, **dict(f)} for state in states for f in _TABLE[i][sentence]
                  if all(state.get(k, v) == v for k, v in f)]
    ok = any((s['setting'], s['problem']) not in EXCLUDED and
             s['twist'] in twists_for(s['problem'], s['item']) for s in states)
    return 'consistent' if ok else 'inconsistent'


def main():
    candidates = list(all_slots())
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    # Take problems and turns in equal turns so none dominates.
    buckets = {}
    for slots in candidates:
        buckets.setdefault((slots['problem'], slots['twist']), []).append(slots)
    keys = sorted(buckets)
    seen, records = set(), []
    while len(records) < COUNT:
        progressed = False
        for key in keys:
            while buckets[key]:
                slots = buckets[key].pop()
                text = ''.join(s for s, _ in compose(**slots))
                if text not in seen and len(text) + 1 <= MAX_CONTEXT:
                    seen.add(text)
                    records.append(dict(slots=slots, text=text))
                    progressed = True
                    break
            if len(records) == COUNT:
                break
        assert progressed, 'not enough distinct stories'

    # Hold out whole (setting, keepsake, problem) combinations: evaluation
    # stories combine known sentences in ways never seen together in training.
    combos = sorted({(r['slots']['setting'], r['slots']['item'], r['slots']['problem'])
                     for r in records})
    def sentences(indices):
        return {s for i in indices for s in split_sentences(records[i]['text'])}

    # Some sentences occur in one combination only (a keepsake's solution
    # carrying a setting's delay); redraw until every one is in training.
    split_rng = random.Random(SPLIT_SEED)
    while True:
        held = set(split_rng.sample(combos, round(len(combos) * 0.15)))
        test_idx = [i for i, r in enumerate(records)
                    if (r['slots']['setting'], r['slots']['item'], r['slots']['problem']) in held]
        test_set = set(test_idx)
        train_idx = [i for i in range(COUNT) if i not in test_set]
        if sentences(test_idx) <= sentences(train_idx):
            break
    assert all(check_story(r['text']) == 'consistent' for r in records)

    specials = ['〈PAD〉', '〈始〉', '〈終〉']
    vocab = specials + sorted(set(''.join(r['text'] for r in records)))
    # The engine needs a vocabulary size divisible by four.
    vocab[len(specials):len(specials)] = [f'〈予備{i + 1}〉' for i in range(-len(vocab) % 4)]
    index = {t: i for i, t in enumerate(vocab)}
    rows = [[1] + [index[c] for c in r['text']] + [2] for r in records]

    corpus_records = []
    for i, r in enumerate(records):
        s = r['slots']
        corpus_records.append(dict(
            id=i + 1, text=r['text'], reading='', group=s['setting'],
            scene=SETTINGS[s['setting']]['scene'], source='generated-sf-story-template',
            split='test' if i in test_set else 'development',
            name=s['name'], item=s['item'], problem=s['problem'], twist=s['twist']))

    examples = train_idx[:2] + test_idx[:2]
    data = dict(
        vocab=vocab,
        texts=[records[i]['text'] for i in examples],
        tokens=[['〈始〉'] + list(records[i]['text']) for i in examples],
        targets=[list(records[i]['text']) + ['〈終〉'] for i in examples],
        exampleSplits=[corpus_records[i]['split'] for i in examples],
        exampleScenes=[corpus_records[i]['scene'] for i in examples],
        rows=rows,
        trainIndices=train_idx,
        testIndices=test_idx,
        corpusRecords=corpus_records,
        corpusMeta=dict(
            generated=COUNT, published=0, scenes=len(SETTINGS),
            testGroups=list(SETTINGS), developmentGroups=list(SETTINGS),
            testCount=len(test_idx), trainCount=len(train_idx),
            maxContext=max(len(row) for row in rows) - 1,
            method='起承転結4文のSF掌編。舞台・主人公・形見・問題・転を選び、後の文が前の文と整合するよう生成。'
                   '転は形見による解決・隠れた原因・見知らぬ存在の3種。形見で解決できる問題は形見ごとに決まる。'
                   '通信遅延・付着物・光は舞台で決まる。人による品質審査は未実施。',
            seed=SEED, splitSeed=SPLIT_SEED,
            splitStrategy='held-out-setting-item-problem-combinations',
            splitMethod='(舞台, 形見, 問題) の組み合わせの15%を評価用に保留。'
                        '評価側の各文・各文字はすべて学習側にも含まれる。'),
    )
    Path(__file__).with_name('corpus-sf.json').write_text(
        json.dumps(data, ensure_ascii=False, separators=(',', ':')))

    lengths = [len(r['text']) for r in records]
    print(f'{COUNT} stories; train {len(train_idx)} / evaluation {len(test_idx)}; '
          f'vocab {len(vocab)}; text length {min(lengths)}-{max(lengths)} '
          f'(mean {sum(lengths) / len(lengths):.1f}); maxContext {data["corpusMeta"]["maxContext"]}')


if __name__ == '__main__':
    main()
