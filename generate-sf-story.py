"""Generate the SF short-story corpus (corpus-sf.json) in the corpus-common.json format.

Each story is four sentences, 起承転結:
  起 introduces the setting, the hero and a keepsake.
  承 brings a problem: lost signal, failing oxygen, a lost way, a blackout,
     contact with an unknown intelligence, or an artifact of advanced technology.
  転 turns it: the keepsake solves it, a hidden cause is revealed,
     or a stranger appears.
  結 resolves it the way that particular turn set up.

Settings range over the solar system, other stars, a future Earth and a
present day where a different science took hold. Later sentences repeat the
hero's name, the keepsake and setting facts (who is at the other end of the
line and how late the answer comes, what covers the antenna, what lights the
way, the facility to return to). Some turns only work in some settings: a
compass needs a magnetic field, and only sealed habitats can run out of air.
check_story() tells whether a story, for example one the model wrote, holds
together.
"""
from collections import Counter
from pathlib import Path
import json
import random

SEED = 2026
SPLIT_SEED = 731
COUNT = 5000
MAX_CONTEXT = 128  # tokens fed to the model: 〈始〉 + text (targets: text + 〈終〉)
TEST_SHARE = 0.15


def setting(scene, place, facility, home, delay, dust, glow, where,
            sealed, surface, magnetic):
    return dict(scene=scene, place=place, facility=facility, home=home, delay=delay,
                dust=dust, glow=glow, where=where,
                sealed=sealed, surface=surface, magnetic=magnetic)


# sealed: the air can run out; surface: one can walk out and get lost;
# magnetic: a compass points somewhere.
SETTINGS = {
    'mars': setting('火星', '火星の基地', '基地', '地球', '四十分後', '赤い砂',
                    '砂嵐の向こうの青い光', '古い溶岩洞の奥', True, True, False),
    'moon': setting('月面', '月面の観測所', '観測所', '地球', '三秒後', '細かい灰色の砂',
                    '地球の照り返し', 'クレーターの底', True, True, False),
    'europa': setting('エウロパ', 'エウロパの氷上基地', '基地', '地球', '二時間後', '厚い霜',
                      '氷の下から差す青い光', '氷の割れ目の底', True, True, False),
    'titan': setting('タイタン', 'タイタンの湖畔基地', '基地', '地球', '三時間後', '黒いもや',
                     '霧の中を漂う光の粒', 'メタンの湖の底', True, True, False),
    'station': setting('軌道ステーション', '軌道ステーション', 'ステーション', '地上', '数秒後',
                       '宇宙のちり', None, '船外のアンテナの陰', True, False, False),
    'proxima': setting('プロキシマb', 'プロキシマbの開拓村', '開拓村', '地球', '八年後', '赤い花粉',
                       '赤い森で光る胞子', '赤い森の奥', True, True, True),
    'trappist': setting('海の惑星', 'トラピストの海の惑星', '浮き島', '地球', '八十年後', '塩の結晶',
                        '海面で光る夜光虫', '浅い海の底', True, True, True),
    'ship': setting('世代船', '星々を渡る世代船', '船', '地球', '二百年後', '白い霜',
                    None, '使われていない貨物室', True, False, False),
    'tokyo': setting('未来の東京', '海に沈んだ未来の東京', '水上の街', '高台の都市', '数分後', '潮の塩',
                     '水没したビルの非常灯', '水没したビルの地下', False, True, True),
    'dome': setting('氷河期の地球', '氷河期の地球のドーム都市', 'ドーム', '隣のドーム', '翌朝', '分厚い氷',
                    '氷の下で光るケーブル', '氷の下の古い地下鉄', False, True, True),
    'osaka': setting('飛行船の大阪', '飛行船が行き交う現代の大阪', '格納庫', '管制塔', '数秒後', '鳩の羽根',
                     '飛行船の誘導灯', '古い格納庫の奥', False, True, True),
    'kyoto': setting('エーテルの京都', 'エーテル通信が広まった現代の京都', '研究所', '本局', '一瞬の後',
                     '古い蜘蛛の巣', '路地に浮かぶエーテルの灯', '寺の地下の蔵', False, True, True),
}

NAMES = ['ミナ', 'ケン', 'ソラ', 'ユウ', 'リコ', 'ハル', 'ノア', 'アキ', 'レイ', 'サキ', 'トオル', 'エマ']

# key: (introduced as, referred to later as)
ITEMS = {
    'radio': ('祖母のラジオ', 'ラジオ'),
    'seed': ('一粒の種', '種'),
    'watch': ('父の時計', '時計'),
    'photo': ('古い写真', '写真'),
    'letter': ('母の手紙', '手紙'),
    'compass': ('古い方位磁石', '方位磁石'),
    'telescope': ('小さな望遠鏡', '望遠鏡'),
    'flashlight': ('手回しの懐中電灯', '懐中電灯'),
    'musicbox': ('古いオルゴール', 'オルゴール'),
}

KI = ['{place}で、{n}は{item}をいつも持ち歩いていた。',
      '{n}は{place}でひとり働き、{item}だけを友にしていた。',
      '{place}に暮らす{n}の宝物は、{item}だった。']

SHO = {
    'signal': ['ある日、{home}との通信が突然途絶えた。',
               '嵐のあと、{home}からの声が聞こえなくなった。'],
    'oxygen': ['ある夜、酸素装置が壊れ、残りは三日分になった。',
               '警報が鳴り、酸素が漏れていると表示された。'],
    'lost': ['外の調査中、{n}は{facility}への道を見失った。',
             '乗り物が止まり、{n}は{facility}から遠く取り残された。'],
    'power': ['ある晩、{facility}の電力が突然すべて止まった。',
              '停電が続き、{facility}の暖房が冷えていった。'],
    'contact': ['{n}は{where}で、規則正しく点滅する光を見つけた。',
                '{where}から、毎晩同じ形の信号が届き始めた。'],
    'artifact': ['{n}は{where}で、誰が作ったかわからない装置を掘り当てた。',
                 '{where}に、まだ温かい金属の箱が埋まっていた。'],
}

# (problem, keepsake) -> (転, 結): the keepsake turns the story.
SOLVES = {
    ('signal', 'radio'): ('{n}はラジオの部品で送信機を組み直した。',
                          '{delay}、{home}から「聞こえるよ」と返事が来た。'),
    ('signal', 'letter'): ('手紙の隅に、古い非常用の周波数が書かれていた。',
                           'その周波数で呼ぶと、{delay}に懐かしい声が返ってきた。'),
    ('oxygen', 'seed'): ('{n}は種を水耕槽にまき、葉に酸素を作らせた。',
                         '{n}は緑の葉に囲まれて、救助の日を待った。'),
    ('oxygen', 'radio'): ('{n}はラジオで近くを通る貨物船を呼んだ。',
                          '三日後、貨物船が{n}を迎えに来た。'),
    ('lost', 'watch'): ('時計の針と星の位置から、{n}は方角を割り出した。',
                        '{n}は時計を握りしめ、夜明け前に{facility}へ戻った。'),
    ('lost', 'photo'): ('写真に写る山の形が、遠くの稜線と重なった。',
                        '{n}は写真の山を目指して歩き、{facility}にたどり着いた。'),
    ('lost', 'compass'): ('方位磁石の針は、迷わず{facility}の方を指していた。',
                          '針を信じて歩き、{n}は夜までに{facility}へ戻った。'),
    ('lost', 'telescope'): ('望遠鏡をのぞくと、地平線に{facility}の灯りが見えた。',
                            '{n}は灯りを目指して歩き、無事に{facility}へ戻った。'),
    ('power', 'flashlight'): ('{n}は懐中電灯の発電機で、非常回路を動かした。',
                              '明かりが戻り、{facility}の機械が一つずつ目を覚ました。'),
    ('contact', 'musicbox'): ('{n}がオルゴールを鳴らすと、光が同じ旋律で応えた。',
                              'その夜から、{n}と光は毎晩歌を交わした。'),
    ('contact', 'radio'): ('ラジオが光の点滅を、知らない言葉の声に変えた。',
                           '{n}は辞書を作り始め、最初の単語は「こんにちは」だった。'),
    ('contact', 'telescope'): ('望遠鏡でのぞくと、光の奥で小さな影が手を振っていた。',
                               '{n}は望遠鏡を下ろし、大きく手を振り返した。'),
    ('artifact', 'watch'): ('時計を近づけると、装置の針が同じ速さで動き出した。',
                            '装置の裏には、時計と同じ職人の名が刻まれていた。'),
    ('artifact', 'photo'): ('写真の隅に、同じ形の装置が写っていた。',
                            '写真の中で装置を抱えていたのは、若い日の祖母だった。'),
    ('artifact', 'letter'): ('手紙の最後に、この装置の図が描かれていた。',
                             '手紙の手順どおりに触れると、装置は静かに目を覚ました。'),
}

# twist -> problem -> [(転, 結)]: each 結 answers its own 転.
TWISTS = {
    'reveal': {
        'signal': [('原因は、アンテナに積もった{dust}だった。',
                    '{n}が{dust}を払うと、{delay}に{home}の声が戻った。'),
                   ('途絶えたのは、{home}側の送信所が壊れたからだった。',
                    '{n}は{ref}を握り、{home}の修理を信じて待ち続けた。')],
        'oxygen': [('だが本当に壊れていたのは、酸素計の方だった。',
                    '酸素はまだ百日分あり、{n}は{ref}を見て笑った。'),
                   ('漏れていたのは、使っていない倉庫の空気だった。',
                    '{n}は倉庫の扉を閉め、{ref}をそっと机に戻した。')],
        'lost': [('足元の足跡は、さっき通った{n}自身のものだった。',
                  '{n}は足跡を逆にたどり、{ref}を握って{facility}に戻った。'),
                 ('道に迷ったのではなく、地図の方が古かった。',
                  '{n}は地図を描き直し、{ref}と一緒に{facility}へ帰った。')],
        'power': [('止まったのは、発電機の点検の時刻だったからだ。',
                   '一時間後に電力は戻り、{n}は{ref}を見て苦笑した。'),
                  ('電力は、誰かが別の場所へこっそり流していた。',
                   'たどった先では、凍えた小さな生き物たちが身を寄せていた。')],
        'contact': [('光の点滅は、素数を一つずつ数えていた。',
                     '{n}は次の素数を光で送り返し、{ref}を胸に返事を待った。'),
                    ('信号は、百年後の{n}自身が送ったものだった。',
                     '{n}は{ref}をその場に埋め、未来の自分への合図にした。')],
        'artifact': [('装置は、百万年前の人類が残したものだった。',
                      '{n}は{ref}と並べて装置を置き、人類の長い歴史を思った。'),
                     ('装置の中には、{n}の名前が書かれた紙が入っていた。',
                      '{n}はその紙に今日の日付を書き足し、装置を元に戻した。')],
    },
    'stranger': {
        'signal': [('そのとき、知らない言葉の信号が届いた。',
                    '{n}は{ref}をそばに置き、初めての返事を送った。'),
                   ('かわりに、{home}ではない誰かが応答した。',
                    '{n}はその誰かと、{home}との通信が戻るまで話し続けた。')],
        'oxygen': [('そのとき、扉の外に見知らぬ影が立っていた。',
                    '影が置いた酸素の箱で、{n}は救助まで生き延びた。'),
                   ('通気口から、知らない誰かが空気を送ってきた。',
                    '{n}は通気口に{ref}を置き、感謝のしるしにした。')],
        'lost': [('{glow}が、{n}の前に道を照らした。',
                  '光を追って{facility}に戻り、{n}は{ref}にそっと礼を言った。'),
                 ('どこからか、{n}の名を呼ぶ声がした。',
                  '声の方へ歩くと{facility}があり、{n}は{ref}を握りしめた。')],
        'power': [('暗闇の中で、壁の向こうから誰かがノックした。',
                   '翌朝、扉の前に{facility}を動かす小さな電池が置かれていた。'),
                  ('見知らぬ機械が現れ、{facility}に電力を分け与えた。',
                   '機械は何も言わずに去り、{n}は{ref}に今日のことを話した。')],
        'contact': [('光の中から、{n}によく似た誰かが現れた。',
                     'その誰かは{ref}を指さし、同じものを持っていると示した。'),
                    ('光は、{n}の言葉を真似して話し始めた。',
                     '{n}は{ref}の話を聞かせ、光はそれを静かに聞いていた。')],
        'artifact': [('装置が開き、中から小さな機械が歩み出た。',
                      '機械は{n}の{ref}を見つめ、同じ形を作って差し出した。'),
                     ('装置から、知らない誰かの声が{n}の名を呼んだ。',
                      '{n}が答えると、装置は星の地図を空中に描いた。')],
    },
}


def problem_allowed(setting_key, problem):
    st = SETTINGS[setting_key]
    if problem == 'oxygen':
        return st['sealed']
    if problem == 'lost':
        return st['surface']
    return True


def arcs(setting_key, problem, item):
    """Every (twist, arc id, 転, 結) available for these slots."""
    st, out = SETTINGS[setting_key], []
    solve = SOLVES.get((problem, item))
    # A compass is useless without a magnetic field.
    if solve and not (item == 'compass' and not st['magnetic']):
        out.append(('item', 'item', *solve))
    for twist, by_problem in TWISTS.items():
        for k, (ten, ketsu) in enumerate(by_problem[problem]):
            if '{glow}' in ten and st['glow'] is None:
                continue
            out.append((twist, f'{twist}{k}', ten, ketsu))
    return out


def compose(setting, name, item, problem, arc, choice):
    """Return [(sentence, facts it implies)] for one choice of every slot."""
    st = SETTINGS[setting]
    ki, sho = choice
    twist, _, ten, ketsu = next(a for a in arcs(setting, problem, item) if a[1] == arc)
    fill = dict(n=name, item=ITEMS[item][0], ref=ITEMS[item][1], **st)

    def facts(template, **base):
        # A sentence pins down the slots its placeholders name.
        if '{n}' in template:
            base['name'] = name
        if '{ref}' in template or base.get('twist') == 'item':
            base['item'] = item
        if any('{' + k + '}' in template for k in
               ('place', 'facility', 'home', 'delay', 'dust', 'glow', 'where')):
            base['setting'] = setting
        return base

    return [
        (KI[ki].format(**fill), dict(setting=setting, name=name, item=item)),
        (SHO[problem][sho].format(**fill), facts(SHO[problem][sho], problem=problem)),
        (ten.format(**fill), facts(ten, problem=problem, twist=twist, arc=arc)),
        (ketsu.format(**fill), facts(ketsu, problem=problem, twist=twist, arc=arc)),
    ]


def all_slots():
    for setting_key in SETTINGS:
        for problem in SHO:
            if not problem_allowed(setting_key, problem):
                continue
            for item in ITEMS:
                for _, arc, _, _ in arcs(setting_key, problem, item):
                    for name in NAMES:
                        for choice in [(k, s) for k in range(len(KI))
                                       for s in range(len(SHO[problem]))]:
                            yield dict(setting=setting_key, name=name, item=item,
                                       problem=problem, arc=arc, choice=choice)


def split_sentences(text):
    parts = text.split('。')
    return [p + '。' for p in parts[:-1]] if parts[-1] == '' else None


_TABLE = None


def check_story(text):
    """Classify a story: 'consistent', 'inconsistent' or 'unknown-sentence'.

    A sentence the generator never produces makes it 'unknown-sentence'.
    Otherwise the story is consistent when one assignment of setting, hero,
    keepsake, problem and turn explains all four sentences and that turn is
    possible there.
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
    ok = any(problem_allowed(s['setting'], s['problem']) and
             s['arc'] in {a[1] for a in arcs(s['setting'], s['problem'], s['item'])}
             for s in states)
    return 'consistent' if ok else 'inconsistent'


def main():
    candidates = list(all_slots())
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    # Take problem × turn buckets in turn so no kind of story dominates.
    buckets = {}
    for slots in candidates:
        twist = 'item' if slots['arc'] == 'item' else slots['arc'][:-1]
        buckets.setdefault((slots['problem'], twist), []).append(slots)
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

    # Hold out whole (setting, keepsake, problem) combinations, so evaluation
    # stories put known sentence patterns together in new ways. A combination
    # is held out only if all its characters stay in training.
    def combo(r):
        return r['slots']['setting'], r['slots']['item'], r['slots']['problem']
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

    def sentences(indices):
        return {s for i in indices for s in split_sentences(records[i]['text'])}
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
            name=s['name'], item=s['item'], problem=s['problem'], twist=s['arc']))

    # Show examples from different settings and problems.
    def pick(indices, count):
        chosen, used = [], set()
        for i in indices:
            key = (records[i]['slots']['setting'], records[i]['slots']['problem'])
            if key[0] not in {k[0] for k in used} and key[1] not in {k[1] for k in used}:
                chosen.append(i)
                used.add(key)
            if len(chosen) == count:
                return chosen
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
            method='起承転結4文のSF掌編。舞台・主人公・形見・問題・転を選び、後の文が前の文と整合するよう生成。'
                   '舞台は太陽系・系外惑星・未来の地球・別の科学が発達した現代の12種。'
                   '問題は通信途絶・酸素・遭難・停電・未知の知性・謎の装置の6種。'
                   '転は形見による解決・隠れた原因・見知らぬ存在で、結は転ごとに決まる。'
                   '方位磁石は磁場のある舞台でのみ、酸素の問題は密閉居住区でのみ起こる。人による品質審査は未実施。',
            seed=SEED, splitSeed=SPLIT_SEED,
            splitStrategy='held-out-setting-item-problem-combinations',
            splitMethod='(舞台, 形見, 問題) の組み合わせを単位に約15%を評価用に保留。'
                        '評価側の文字はすべて学習側にも含まれる。文の型は共通で、'
                        '名前・舞台・形見の入った具体的な文は評価側にだけ現れることがある。'),
    )
    Path(__file__).with_name('corpus-sf.json').write_text(
        json.dumps(data, ensure_ascii=False, separators=(',', ':')))

    lengths = [len(r['text']) for r in records]
    print(f'{COUNT} stories from {len(candidates):,} possible; train {len(train_idx)} / '
          f'evaluation {len(test_idx)}; vocab {len(vocab)}; text length '
          f'{min(lengths)}-{max(lengths)} (mean {sum(lengths) / len(lengths):.1f}); '
          f'maxContext {data["corpusMeta"]["maxContext"]}; '
          f'distinct sentences {len(sentences(range(COUNT)))}')


if __name__ == '__main__':
    main()
