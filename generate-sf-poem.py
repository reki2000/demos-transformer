"""Generate the SF poem corpus (corpus-sf.json) in the corpus-common.json format.

Each poem has seven lines joined by '／':
  1. opening   : depends on the setting
  2. detail    : another fact of the same setting
  3. intro     : introduces the subject and the keepsake (motif)
  4. event     : depends on the event and the setting (signal delay, discovery)
  5. reaction  : repeats the subject; for 'return', names the setting's view
  6. closing   : repeats the motif; its mood follows the event
  7. coda      : its mood follows the event

So later lines are fixed by earlier ones, and check_poem() can tell
whether a poem (for example, one the model generated) is consistent.
"""
from pathlib import Path
import json
import random

SEED = 2026
SPLIT_SEED = 731
COUNT = 5000
MAX_CONTEXT = 128  # tokens fed to the model: 〈始〉 + text (targets: text + 〈終〉)
SEP = '／'

SETTINGS = {
    'mars': dict(
        scene='火星',
        openings=['赤い砂の地平に青い夕日が沈む', '火星の風が錆びた基地を鳴らす'],
        details=['重さは地球の三分の一ほどしかない', '昼でも空は薄い砂の色をしている'],
        delay='二十分',
        discovery='砂の下に凍った水が眠っていた',
        view='赤い空'),
    'moon': dict(
        scene='月面',
        openings=['黒い空に青い地球が浮かんでいる', '音のない月の野に影が長く伸びる'],
        details=['一歩ごとに体が軽く浮き上がる', '昼は二週間続き夜も二週間続く'],
        delay='一秒',
        discovery='古い足跡がまだ消えずに残っていた',
        view='黒い空'),
    'station': dict(
        scene='軌道ステーション',
        openings=['窓の外を地球の夜が流れていく', '九十分ごとに朝が来る軌道の上'],
        details=['浮かんだ水の粒がゆっくり回る', '一日に十六回も日が昇り沈む'],
        delay='一瞬',
        discovery='窓の外を見知らぬ船が横切った',
        view='青い地球'),
    'europa': dict(
        scene='エウロパ',
        openings=['凍った海の上に大きな木星が昇る', '氷の割れ目から白い霧が立つ'],
        details=['氷の下には深い海が眠っている', '木星の光で氷が淡く照らされる'],
        delay='一時間',
        discovery='氷の下で青い光がゆっくり瞬いた',
        view='木星'),
    'titan': dict(
        scene='タイタン',
        openings=['橙色の霧の中にメタンの雨が降る', '土星の輪が霧の向こうにかすむ'],
        details=['地球よりも濃い大気が体を包む', '湖を満たすのは水ではなくメタンだ'],
        delay='一時間半',
        discovery='霧の湖を何かの影が渡っていった',
        view='霧の空'),
    'ship': dict(
        scene='世代船',
        openings=['千年の航海の途中で船は闇を進む', '星のない窓の奥で船は静かに眠る'],
        details=['祖先の顔を知る者はもういない', '船の中だけで人は生まれ死んでいく'],
        delay='百年',
        discovery='進む先に見たことのない星が灯った',
        view='星の海'),
}

SUBJECTS = ['僕', '私', '彼女', '少年', '老いた技師', '最後のロボット']

INTROS = ['{s}は{m}を握っている', '{s}の胸には{m}がある', '{s}は{m}を見つめている']

# event -> (mood, event lines, reaction lines)
EVENTS = {
    'signal': ('hope',
               ['地球の声が{delay}遅れて届いた'],
               ['{s}は「聞こえるよ」と答えた', '{s}はその声に耳をすます']),
    'silence': ('lonely',
                ['{delay}待っても地球から返事は来ない'],
                ['{s}はひとりで窓辺に座る', '{s}は送信を何度も繰り返す']),
    'crisis': ('resolve',
               ['残りの酸素はあと三日分', '燃料計の針が赤に近づく'],
               ['{s}は静かに計算をやり直す', '{s}は眠らずに修理を続ける']),
    'return': ('farewell',
               ['地球から帰還の命令が届いた', '帰りの船の準備ができた'],
               ['{s}は{view}を最後に見つめた']),
    'discovery': ('wonder',
                  ['{discovery}'],
                  ['{s}は息を止めて見つめた', '{s}は記録の手を止めた']),
}
# A generation ship never returns.
EXCLUDED = {('ship', 'return')}

MOTIFS = {
    'watch': dict(name='母の時計', closings=dict(
        hope='母の時計が地球の朝を告げている',
        lonely='母の時計だけが地球の時を刻む',
        resolve='母の時計を巻いて明日を待つ',
        farewell='母の時計を地球の時刻に戻した',
        wonder='母の時計の針がその瞬間で止まった')),
    'seed': dict(name='一粒の種', closings=dict(
        hope='いつかこの種を蒔く場所がある',
        lonely='手の中の種はまだ眠ったままだ',
        resolve='種だけは必ず未来に届ける',
        farewell='種を一粒ここに残していく',
        wonder='手の中の種がかすかに温かい')),
    'photo': dict(name='古い写真', closings=dict(
        hope='写真の中の海が少し近くなる',
        lonely='写真の中の人はもう年を取らない',
        resolve='写真を胸にしまい扉を閉めた',
        farewell='写真をこの場所に置いて帰る',
        wonder='写真の裏にこの景色を書き足した')),
    'radio': dict(name='壊れたラジオ', closings=dict(
        hope='壊れたラジオがふいに歌い出した',
        lonely='ラジオは今夜も雑音だけを流す',
        resolve='ラジオの部品で送信機を組み直す',
        farewell='ラジオの電源をそっと切った',
        wonder='ラジオが知らない周波数を拾った')),
    'stone': dict(name='地球の石', closings=dict(
        hope='地球の石を握ると海の音がした',
        lonely='地球の石は冷たく黙っている',
        resolve='地球の石を重しにして日誌を書く',
        farewell='地球の石を元の場所へ返しに行く',
        wonder='地球の石がかすかに震えた')),
    'letter': dict(name='紙の手紙', closings=dict(
        hope='手紙の返事をやっと書き始める',
        lonely='手紙はまだ封を開けられずにいる',
        resolve='手紙の最後の行をもう一度読む',
        farewell='手紙を持って地球へ帰ろう',
        wonder='手紙に書く言葉がまた一つ増えた')),
}


CODAS = dict(
    hope=['遠い場所はもう遠くない', '明日の空はきっと明るい'],
    lonely=['宇宙はただ広く静かだ', '誰も知らない夜がまた来る'],
    resolve=['まだ終わるわけにはいかない', '生き延びるために手を動かす'],
    farewell=['さよならを小さくつぶやいた', 'もう二度とここには来ない'],
    wonder=['宇宙はまだ答えを隠している', 'この夜を誰かに伝えたい'],
)


def compose(setting, subject, motif, event, choice):
    """Return [(line, facts it implies)] for one choice of every slot.

    choice holds the variant index of each line: opening, detail, intro,
    event, reaction, coda.
    """
    st, mo = SETTINGS[setting], MOTIFS[motif]
    mood, events, reactions = EVENTS[event]
    o, d, i, e, r, c = choice
    fill = dict(s=subject, m=mo['name'], delay=st['delay'],
                discovery=st['discovery'], view=st['view'])
    event_line = events[e].format(**fill)
    return [
        (st['openings'][o], dict(setting=setting)),
        (st['details'][d], dict(setting=setting)),
        (INTROS[i].format(**fill), dict(subject=subject, motif=motif)),
        # Only lines naming a delay or a discovery depend on the setting.
        (event_line, dict(event=event, **({'setting': setting} if '{' in events[e] else {}))),
        (reactions[r].format(**fill),
         dict(subject=subject, event=event, **({'setting': setting} if '{view}' in reactions[r] else {}))),
        (mo['closings'][mood], dict(motif=motif, mood=mood)),
        (CODAS[mood][c], dict(mood=mood)),
    ]


def combinations():
    """Every allowed (setting, event) pair."""
    return [(st, ev) for st in SETTINGS for ev in EVENTS if (st, ev) not in EXCLUDED]


def random_slots(rng):
    """Pick slots with the event uniform first, so every event is equally common."""
    event = rng.choice(list(EVENTS))
    setting = rng.choice([st for st, ev in combinations() if ev == event])
    st, (_, events, reactions) = SETTINGS[setting], EVENTS[event]
    choice = (rng.randrange(len(st['openings'])), rng.randrange(len(st['details'])),
              rng.randrange(len(INTROS)), rng.randrange(len(events)),
              rng.randrange(len(reactions)), rng.randrange(2))
    return dict(setting=setting, subject=rng.choice(SUBJECTS),
                motif=rng.choice(list(MOTIFS)), event=event, choice=choice)


def _line_table():
    """Map every canonical line to the fact sets it can imply, per position."""
    table = [dict() for _ in range(7)]
    for setting, event in combinations():
        st, (_, events, reactions) = SETTINGS[setting], EVENTS[event]
        for motif in MOTIFS:
            for subject in SUBJECTS:
                for choice in [(o, d, i, e, r, c) for o in range(2) for d in range(2)
                               for i in range(len(INTROS)) for e in range(len(events))
                               for r in range(len(reactions)) for c in range(2)]:
                    for pos, (line, facts) in enumerate(
                            compose(setting, subject, motif, event, choice)):
                        table[pos].setdefault(line, set()).add(tuple(sorted(facts.items())))
    return [{line: [dict(f) for f in facts] for line, facts in pos.items()} for pos in table]


_TABLE = None


def check_poem(text):
    """Classify a poem: 'consistent', 'inconsistent' or 'unknown-line'.

    A line the generator never produces makes the poem 'unknown-line'.
    Otherwise the poem is consistent when one assignment of setting,
    subject, motif, event and mood explains every line.
    """
    global _TABLE
    if _TABLE is None:
        _TABLE = _line_table()
    lines = text.split(SEP)
    if len(lines) != len(_TABLE) or any(line not in _TABLE[i] for i, line in enumerate(lines)):
        return 'unknown-line'
    states = [{}]
    for i, line in enumerate(lines):
        states = [{**state, **facts} for state in states for facts in _TABLE[i][line]
                  if all(state.get(k, v) == v for k, v in facts.items())]
    consistent = any(EVENTS[s['event']][0] == s['mood'] and
                     (s['setting'], s['event']) not in EXCLUDED for s in states)
    return 'consistent' if consistent else 'inconsistent'


def main():
    rng = random.Random(SEED)
    seen, records = set(), []
    while len(records) < COUNT:
        slots = random_slots(rng)
        text = SEP.join(line for line, _ in compose(**slots))
        if text in seen or len(text) + 1 > MAX_CONTEXT:
            continue
        seen.add(text)
        records.append(dict(slots=slots, text=text))

    # Hold out whole (setting, motif, event) combinations: evaluation poems
    # combine known lines in ways never seen together during training.
    combos = sorted({(r['slots']['setting'], r['slots']['motif'], r['slots']['event'])
                     for r in records})
    split_rng = random.Random(SPLIT_SEED)
    held = set(split_rng.sample(combos, round(len(combos) * 0.15)))
    test_idx = [i for i, r in enumerate(records)
                if (r['slots']['setting'], r['slots']['motif'], r['slots']['event']) in held]
    train_idx = [i for i in range(COUNT) if i not in set(test_idx)]

    train_lines = {line for i in train_idx for line in records[i]['text'].split(SEP)}
    test_lines = {line for i in test_idx for line in records[i]['text'].split(SEP)}
    assert test_lines <= train_lines, 'evaluation-only line'
    assert set(''.join(records[i]['text'] for i in test_idx)) <= \
        set(''.join(records[i]['text'] for i in train_idx)), 'evaluation-only character'
    assert all(check_poem(r['text']) == 'consistent' for r in records)

    specials = ['〈PAD〉', '〈始〉', '〈終〉']
    vocab = specials + sorted(set(''.join(r['text'] for r in records)))
    index = {t: i for i, t in enumerate(vocab)}
    rows = [[1] + [index[c] for c in r['text']] + [2] for r in records]
    assert max(len(row) for row in rows) - 1 <= MAX_CONTEXT

    corpus_records = []
    test_set = set(test_idx)
    for i, r in enumerate(records):
        s = r['slots']
        corpus_records.append(dict(
            id=i + 1, text=r['text'], reading='', group=s['setting'],
            scene=SETTINGS[s['setting']]['scene'], source='generated-sf-template',
            split='test' if i in test_set else 'development',
            subject=s['subject'], motif=s['motif'], event=s['event'],
            mood=EVENTS[s['event']][0]))

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
            method='5行のSF詩。舞台・主体・形見・出来事を選び、後の行が前の行と整合するよう生成。'
                   '通信遅延と発見は舞台、反応の主体は2行目、結びの形見と情感は2行目と出来事で決まる。'
                   '人による詩としての品質審査は未実施。',
            seed=SEED, splitSeed=SPLIT_SEED,
            splitStrategy='held-out-setting-motif-event-combinations',
            splitMethod='(舞台, 形見, 出来事) の組み合わせの15%を評価用に保留。'
                        '評価側の各行・各文字はすべて学習側にも含まれる。'),
    )
    path = Path(__file__).with_name('corpus-sf.json')
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))

    lengths = [len(r['text']) for r in records]
    print(f'{COUNT} poems; train {len(train_idx)} / evaluation {len(test_idx)}; '
          f'vocab {len(vocab)}; text length {min(lengths)}-{max(lengths)} '
          f'(mean {sum(lengths) / len(lengths):.1f}); maxContext {data["corpusMeta"]["maxContext"]}')
    for text in data['texts']:
        print(' ', text)


if __name__ == '__main__':
    main()
