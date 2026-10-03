"""Generate the SF short-story corpus (corpus-sf.json) in the corpus-common.json format.

Each story is a 『title』 followed by four sentences, 起承転結:
  起 introduces the setting, the hero and a foreshadowed element (伏線):
     a person, a creature, a machine or an object kept close.
  承 brings a problem: a lost signal, thinning air, a lost way, a blackout,
     a blinking light from something unknown, or a buried device.
  転 turns it: the element holds the answer, turns out to be the culprit,
     reveals its true identity, the day is a time loop, the hero is not
     human, a stranger arrives, or a hidden cause is found.
  結 pays the turn off, usually through the same element.

Stories favor surprise over realism: any problem can happen anywhere.
What they keep is dependency on earlier text, which is what the attention
maps show: the title names the turn, and later sentences bring back the
hero's name, the element and the setting's details (who answers the call
and how late, where things are found, what piles up on the machines).
"""
from collections import Counter
from pathlib import Path
import json
import random
import re

SEED = 2026
SPLIT_SEED = 731
COUNT = 5000
MAX_CONTEXT = 128  # tokens fed to the model: 〈始〉 + text (targets: text + 〈終〉)
TEST_SHARE = 0.15


def setting(scene, place, facility, home, delay, dust, where):
    return dict(scene=scene, place=place, facility=facility, home=home,
                delay=delay, dust=dust, where=where)


SETTINGS = {
    'mars': setting('火星', '火星の基地', '基地', '地球', '四十分後', '赤い砂', '古い溶岩洞の奥'),
    'moon': setting('月面', '月面の観測所', '観測所', '地球', '三秒後', '灰色の砂', 'クレーターの底'),
    'europa': setting('エウロパ', 'エウロパの氷上基地', '基地', '地球', '二時間後', '厚い霜', '氷の割れ目の底'),
    'titan': setting('タイタン', 'タイタンの湖畔基地', '基地', '地球', '三時間後', '黒いもや', 'メタンの湖の底'),
    'station': setting('軌道', '軌道ステーション', 'ステーション', '地上', '数秒後', '宇宙のちり', '船外のアンテナの陰'),
    'proxima': setting('プロキシマ', 'プロキシマbの開拓村', '開拓村', '地球', '八年後', '赤い花粉', '赤い森の奥'),
    'trappist': setting('海の惑星', 'トラピストの海の惑星', '浮き島', '地球', '八十年後', '塩の結晶', '浅い海の底'),
    'ship': setting('世代船', '星々を渡る世代船', '船', '地球', '二百年後', '白い霜', '使われていない貨物室'),
    'tokyo': setting('沈んだ東京', '海に沈んだ未来の東京', '水上の街', '高台の都市', '数分後', '潮の塩', '水没したビルの地下'),
    'dome': setting('ドーム都市', '氷河期のドーム都市', 'ドーム', '隣のドーム', '翌朝', '分厚い氷', '氷の下の古い地下鉄'),
    'osaka': setting('大阪の空', '飛行船の飛ぶ現代の大阪', '格納庫', '管制塔', '数秒後', '鳩の羽根', '古い格納庫の奥'),
    'kyoto': setting('京都', 'エーテル通信の京都', '研究所', '本局', '一瞬の後', '古い蜘蛛の巣', '寺の地下の蔵'),
}

NAMES = ['ミナ', 'ケン', 'ソラ', 'ユウ', 'リコ', 'ハル', 'ノア', 'アキ', 'レイ', 'サキ', 'トオル', 'エマ']

# The foreshadowed element: (introduced as, referred to later as, verb of being).
CLUES = {
    'colleague': ('無口な同僚', '同僚', 'いた'),
    'mechanic': ('年老いた整備士', '整備士', 'いた'),
    'cat': ('迷い込んだ猫', '猫', 'いた'),
    'robot': ('掃除ロボット', 'ロボット', 'いた'),
    'ai': ('名前のない人工知能', '人工知能', 'いた'),
    'radio': ('祖母のラジオ', 'ラジオ', 'あった'),
    'photo': ('古い写真', '写真', 'あった'),
    'seed': ('一粒の種', '種', 'あった'),
    'watch': ('父の時計', '時計', 'あった'),
    'stone': ('小さな青い石', '青い石', 'あった'),
}

TITLE = '『{}』'

KI = ['{place}で働く{n}のそばには、いつも{clue}が{exist}。',
      '{place}に暮らす{n}は、{clue}を何より大切にしていた。',
      '{n}は{place}で、{clue}と暮らしていた。']

# problem -> (承 variants, how it ends well, what a culprit was doing, title words)
PROBLEMS = {
    'signal': (['ある日、{home}との通信が突然途絶えた。',
                '嵐のあと、{home}からの声が聞こえなくなった。'],
               '{delay}に{home}の声が戻った', '通信を止めていた', '途切れた声'),
    'air': (['ある夜、{facility}の空気が少しずつ薄くなり始めた。',
             '警報が鳴り、{facility}の空気が漏れていると告げた。'],
            '{facility}に新しい空気が流れ込んだ', '空気を吸い込んでいた', '薄い空気'),
    'lost': (['外の調査中、{n}は{facility}への道を見失った。',
              '乗り物が止まり、{n}は{facility}から遠く取り残された。'],
             '{n}は無事に{facility}へ帰り着いた', '道しるべを隠していた', '帰り道'),
    'power': (['ある晩、{facility}の電力が突然すべて止まった。',
               '停電が続き、{facility}は少しずつ冷えていった。'],
              '{facility}に明かりが戻った', '電力を使い込んでいた', '消えた明かり'),
    'contact': (['{n}は{where}で、規則正しく点滅する光を見つけた。',
                 '{where}から、毎晩同じ形の信号が届き始めた。'],
                '光は{n}に初めての返事をくれた', 'その光を灯していた', '点滅する光'),
    'artifact': (['{n}は{where}で、誰が作ったかわからない装置を掘り当てた。',
                  '{where}に、まだ温かい金属の箱が埋まっていた。'],
                 '装置は静かに目を覚ました', 'その装置を埋めた', '温かい箱'),
}

# What the element turns out to be: (as revealed, as referred to in 結, title).
IDENTITIES = [
    ('百年前に送られた観測機', '観測機', '百年前の観測機'),
    ('未来から来た{n}自身', '未来の{n}', '未来の自分'),
    ('遠い星から来た使者', '使者', '星からの使者'),
    ('この星そのものの意識', '星の意識', '星の意識'),
    ('{home}が送った見張り', '見張り', '{home}の見張り'),
]

# twist -> [(title, 転, 結)]. {fix} is how the problem ends, {cause} what a
# culprit was doing; {id}, {id_ref} and {id_title} come from IDENTITIES.
TWISTS = {
    'key': [('{ref}が知っていた',
             '{n}は、{ref}に答えが隠されていると気づいた。',
             '{ref}のおかげで、{fix}。')],
    'culprit': [('犯人は{ref}',
                 '{cause}のは、{ref}だった。',
                 '{ref}にも、{n}に気づいてほしい理由があったのだ。'),
                ('{ref}のいたずら',
                 '{cause}のは、なんと{ref}だった。',
                 '{n}が{ref}と話をつけると、{fix}。')],
    'identity': [('{id_title}',
                  '{ref}の正体は、{id}だった。',
                  '{id_ref}に導かれ、{fix}。')],
    'loop': [('{ptitle}をもう一度',
              '{n}は、同じ一日を何度も繰り返していた。',
              '今度こそ{fix}と、{n}は{ref}に告げた。')],
    'android': [('{ref}だけが知る秘密',
                 '{n}は、自分の手が金属だと気づいた。',
                 '{fix}が、{n}の正体は{ref}だけが知っている。')],
    'stranger': [('{scene}の訪問者',
                  '{where}から、見知らぬ誰かが現れた。',
                  '誰かは{ref}を見てうなずき、やがて{fix}。')],
    # Something piled up on the machines only explains machine trouble.
    'dust': [('{scene}の{dust}',
              '原因は、{facility}に積もった{dust}だった。',
              '{n}と{ref}が{dust}を払うと、{fix}。')],
}
DUST_PROBLEMS = {'signal', 'air', 'power'}


def arcs(problem):
    """Every (twist, arc id) possible for a problem."""
    out = []
    for twist, variants in TWISTS.items():
        if twist == 'dust' and problem not in DUST_PROBLEMS:
            continue
        if twist == 'identity':
            out += [(twist, f'identity{k}') for k in range(len(IDENTITIES))]
        else:
            out += [(twist, f'{twist}{k}') for k in range(len(variants))]
    return out


def compose(setting, name, clue, problem, arc, choice):
    """Return [title, 起, 承, 転, 結] for one choice of every slot."""
    st, (intro, ref, exist) = SETTINGS[setting], CLUES[clue]
    sho_variants, fix, cause, ptitle = PROBLEMS[problem]
    twist = arc.rstrip('0123456789')
    k = int(arc[len(twist):])
    title, ten, ketsu = TWISTS[twist][0 if twist == 'identity' else k]
    fill = dict(n=name, clue=intro, ref=ref, exist=exist, cause=cause, ptitle=ptitle, **st)
    if twist == 'identity':
        ident, id_ref, id_title = (s.format(**fill) for s in IDENTITIES[k])
        fill.update(id=ident, id_ref=id_ref, id_title=id_title)
    fill['fix'] = fix.format(**fill)
    ki, sho = choice
    return [TITLE.format(title.format(**fill)), KI[ki].format(**fill),
            sho_variants[sho].format(**fill), ten.format(**fill), ketsu.format(**fill)]


def all_slots():
    for setting_key in SETTINGS:
        for problem in PROBLEMS:
            for _, arc in arcs(problem):
                for clue in CLUES:
                    for name in NAMES:
                        for choice in [(k, s) for k in range(len(KI)) for s in range(2)]:
                            yield dict(setting=setting_key, name=name, clue=clue,
                                       problem=problem, arc=arc, choice=choice)


def split_story(text):
    """Split into [title, 起, 承, 転, 結], or None when the shape is wrong."""
    m = re.fullmatch(r'(『[^』]*』)((?:[^。]*。){4})', text)
    return [m.group(1)] + [s + '。' for s in m.group(2).split('。')[:-1]] if m else None


# Words that only stories about one problem use.
PROBLEM_WORDS = {'通信': 'signal', '声': 'signal', '空気': 'air', '道を見失': 'lost', '道しるべ': 'lost', '帰り': 'lost', '取り残': 'lost',
                 '電力': 'power', '停電': 'power', '明かり': 'power', '光': 'contact',
                 '信号': 'contact', '装置': 'artifact', '箱': 'artifact'}


def _markers():
    """Words that point at exactly one choice of name, element or setting."""
    out = {'name': {n: n for n in NAMES}, 'clue': {}, 'setting': {}, 'problem': PROBLEM_WORDS}
    for key, (_, ref, _) in CLUES.items():
        out['clue'][ref] = key
    counts = Counter(v for st in SETTINGS.values()
                     for f in ('delay', 'dust', 'where', 'home') for v in [st[f]])
    for key, st in SETTINGS.items():
        for f in ('delay', 'dust', 'where', 'home', 'place'):
            if counts[st[f]] == 1 or f == 'place':
                out['setting'][st[f]] = key
    return out


MARKERS = _markers()


def check_story(text):
    """Loosely check that a story holds together: 'consistent', 'inconsistent' or 'malformed'.

    'malformed' unless it is a 『title』 and four sentences. Otherwise every
    hero name, element and setting detail mentioned anywhere must be the one
    the opening sentence (起) introduced, and every word tied to a problem must
    belong to the problem 承 raised. Wording is not checked, so a model's own
    phrasing passes as long as it does not contradict what came first.
    """
    parts = split_story(text)
    if not parts:
        return 'malformed'
    for kind, words in MARKERS.items():
        # Longest words first, so '青い石' is not also read as a shorter word.
        ordered = sorted(words, key=len, reverse=True)
        def found(s):
            if kind == 'problem':
                # Place names such as エーテル通信の京都 are not about the problem.
                for w in MARKERS['setting']:
                    s = s.replace(w, '')
            hits = set()
            for w in ordered:
                if w in s:
                    hits.add(words[w])
                    s = s.replace(w, '')
            return hits
        first = 2 if kind == 'problem' else 1
        introduced = found(parts[first])
        if len(introduced) != 1 and kind != 'setting':
            return 'inconsistent'
        if kind == 'setting' and not introduced:
            continue
        rest = found(''.join(p for i, p in enumerate(parts) if i != first))
        if rest - introduced:
            return 'inconsistent'
    return 'consistent'


def main():
    candidates = list(all_slots())
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    # Take problem × turn buckets in turn so no kind of story dominates.
    buckets = {}
    for slots in candidates:
        buckets.setdefault((slots['problem'], slots['arc'].rstrip('0123456789')), []).append(slots)
    keys = sorted(buckets)
    seen, records = set(), []
    while len(records) < COUNT:
        progressed = False
        for key in keys:
            while buckets[key]:
                slots = buckets[key].pop()
                text = ''.join(compose(**slots))
                if text not in seen and len(text) + 1 <= MAX_CONTEXT:
                    seen.add(text)
                    records.append(dict(slots=slots, text=text))
                    progressed = True
                    break
            if len(records) == COUNT:
                break
        assert progressed, 'not enough distinct stories'

    # Hold out whole (setting, element, problem) combinations, so evaluation
    # stories put known sentence patterns together in new ways. A combination
    # is held out only if all its characters stay in training.
    def combo(r):
        return r['slots']['setting'], r['slots']['clue'], r['slots']['problem']
    by_combo = {}
    for i, r in enumerate(records):
        by_combo.setdefault(combo(r), []).append(i)
    train_count = Counter(c for r in records for c in r['text'])
    order = sorted(by_combo)
    random.Random(SPLIT_SEED).shuffle(order)
    held, target = [], round(COUNT * TEST_SHARE)
    for key in order:
        if sum(len(by_combo[k]) for k in held) >= target:
            break
        uses = Counter(c for i in by_combo[key] for c in records[i]['text'])
        if all(train_count[c] > n for c, n in uses.items()):
            train_count -= uses
            held.append(key)
    test_set = {i for key in held for i in by_combo[key]}
    test_idx = sorted(test_set)
    train_idx = [i for i in range(COUNT) if i not in test_set]
    assert set(''.join(records[i]['text'] for i in test_idx)) <= \
        set(''.join(records[i]['text'] for i in train_idx)), 'evaluation-only character'
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
            name=s['name'], clue=s['clue'], problem=s['problem'], twist=s['arc']))

    # Show examples with different settings, problems and turns.
    def pick(indices, count):
        chosen, used = [], [set(), set(), set()]
        for i in indices:
            s = records[i]['slots']
            key = (s['setting'], s['problem'], s['arc'].rstrip('0123456789'))
            if all(k not in u for k, u in zip(key, used)):
                chosen.append(i)
                for k, u in zip(key, used):
                    u.add(k)
            if len(chosen) == count:
                break
        return chosen
    examples = pick(train_idx, 3) + pick(test_idx, 2)
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
            testGroups=sorted({combo(records[i])[0] for i in test_idx}),
            developmentGroups=list(SETTINGS),
            testCount=len(test_idx), trainCount=len(train_idx),
            maxContext=max(len(row) for row in rows) - 1,
            method='『タイトル』と起承転結4文のSF掌編。起で伏線となる要素（人物・生き物・機械・物）を置き、'
                   '転と結で回収する。転は伏線が鍵・伏線が犯人・正体・時間ループ・主人公が機械・訪問者・積もった塵の7種。'
                   '現実性の制約は設けず、タイトル・名前・伏線・舞台の細部が前の文に依存することを優先。人による品質審査は未実施。',
            seed=SEED, splitSeed=SPLIT_SEED,
            splitStrategy='held-out-setting-clue-problem-combinations',
            splitMethod='(舞台, 伏線, 問題) の組み合わせを単位に約15%を評価用に保留。'
                        '評価側の文字はすべて学習側にも含まれる。'),
    )
    Path(__file__).with_name('corpus-sf.json').write_text(
        json.dumps(data, ensure_ascii=False, separators=(',', ':')))

    lengths = [len(r['text']) for r in records]
    print(f'{COUNT} stories from {len(candidates):,} possible; train {len(train_idx)} / '
          f'evaluation {len(test_idx)}; vocab {len(vocab)}; text length '
          f'{min(lengths)}-{max(lengths)} (mean {sum(lengths) / len(lengths):.1f}); '
          f'maxContext {data["corpusMeta"]["maxContext"]}')


if __name__ == '__main__':
    main()
