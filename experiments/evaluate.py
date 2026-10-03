"""Summarize experiments/results/*.json as Markdown tables (experiments/depth-heads.md).

For stories generated from evaluation titles it reports:
  - consistent / inconsistent / malformed, from check_story() in generate-sf-story.py
  - whether the turn (転) the story takes is the one its title names
  - how many generated stories are copies of a training story
"""
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('story', ROOT / 'generate-sf-story.py')
story = importlib.util.module_from_spec(spec)
spec.loader.exec_module(story)

corpus = json.loads((ROOT / 'corpus-sf.json').read_text())
train_texts = {corpus['corpusRecords'][i]['text'] for i in corpus['trainIndices']}

IDENTITY_TITLES = ('百年前の観測機', '未来の自分', '星からの使者', '星の意識', 'の見張り')


def title_turn(title):
    for words, turn in [(('が知っていた',), 'key'), (('犯人は', 'のいたずら'), 'culprit'),
                        (('をもう一度',), 'loop'), (('だけが知る秘密',), 'android'),
                        (('の訪問者',), 'stranger'), (IDENTITY_TITLES, 'identity')]:
        if any(w in title for w in words):
            return turn
    return 'dust'


def story_turn(body):
    """The turn a story takes, read from its third sentence (転)."""
    sentences = body.split('。')
    ten = sentences[2] if len(sentences) > 2 else ''
    for word, turn in [('答えが隠されている', 'key'), ('正体は', 'identity'), ('繰り返して', 'loop'),
                       ('金属だと', 'android'), ('見知らぬ誰か', 'stranger'), ('積もった', 'dust'),
                       ('ていたのは', 'culprit'), ('でいたのは', 'culprit'), ('埋めたのは', 'culprit')]:
        if word in ten:
            return turn
    return None


def summarize(result):
    gens = result['generations']
    details = [story.check_story_detail(g['text']) for g in gens]
    verdicts = [v for v, _ in details]
    reasons = {k: sum(kind == k for _, kind in details) / len(gens) for k in ('name', 'clue', 'setting', 'problem')}
    turn_ok = 0
    for g in gens:
        title, _, body = g['text'].partition('』')
        turn_ok += story_turn(body) == title_turn(title + '』')
    n = len(gens)
    return dict(
        consistent=verdicts.count('consistent') / n,
        inconsistent=verdicts.count('inconsistent') / n,
        malformed=verdicts.count('malformed') / n,
        turn=turn_ok / n,
        copied=sum(g['text'] in train_texts for g in gens) / n,
        distinct=len({g['text'] for g in gens}) / n,
        reasons=reasons,
    )


def main():
    results = [json.loads(p.read_text()) for p in sorted((ROOT / 'experiments/results').glob('*.json'))]
    results.sort(key=lambda r: (r['dimension'], r['layers'], r['heads']))
    pct = lambda x: f'{x * 100:.0f}%'
    lines = ['# ブロック数とヘッド数の比較（SF掌編）', '',
             f'データ：corpus-sf.json（学習{len(corpus["trainIndices"])}話・評価{len(corpus["testIndices"])}話）。'
             'バッチ16・学習率0.003・4周・seed 42。生成は評価用の各話のタイトルまでを入力し、貪欲法で〈終〉まで（最大128文字）。', '',
             '## 損失と生成の質', '',
             '| 構成 | パラメータ | 評価損失 | 次文字正解率 | 整合 | 不整合 | 形式崩れ | タイトルどおりの転 | 学習データの丸写し | 異なる話の割合 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        s = summarize(r)
        lines.append(f"| {r['layers']}層・{r['dimension']}次元・{r['heads']}ヘッド | {r['parameters']:,} | {r['testLoss']:.3f} | "
                     f"{pct(r['testAccuracy'])} | {pct(s['consistent'])} | {pct(s['inconsistent'])} | {pct(s['malformed'])} | "
                     f"{pct(s['turn'])} | {pct(s['copied'])} | {pct(s['distinct'])} |")
    lines += ['', '## 不整合の内訳', '', '不整合と判定された話が、最初に食い違った要素。割合は生成した話全体に対するもの。', '',
              '| 構成 | 主人公の名前 | 伏線の要素 | 舞台の細部 | 問題 |', '|---|---:|---:|---:|---:|']
    for r in results:
        rs = summarize(r)['reasons']
        lines.append(f"| {r['layers']}層・{r['dimension']}次元・{r['heads']}ヘッド | " + ' | '.join(pct(rs[k]) for k in ('name', 'clue', 'setting', 'problem')) + ' |')
    lines += ['', '## 層ごとの注意の平均距離', '',
              '評価用64話で、各位置の注意が平均何文字前を参照しているか（全ヘッド平均）。括弧内はヘッドごとの最小〜最大。'
              '「20字以上前」は注意の重みのうち20文字以上前に向かう割合。', '',
              '| 構成 | ' + ' | '.join(f'ブロック{b + 1}' for b in range(6)) + ' |',
              '|---|' + '---:|' * 6]
    for r in results:
        cells = []
        for b in range(6):
            if b >= r['layers']:
                cells.append('')
                continue
            heads = r['attention'][b]
            mean = sum(h['distance'] for h in heads) / len(heads)
            far = sum(h['far'] for h in heads) / len(heads)
            span = f" ({min(h['distance'] for h in heads):.1f}〜{max(h['distance'] for h in heads):.1f})" if len(heads) > 1 else ''
            cells.append(f'{mean:.1f}字{span}<br>20字以上前 {pct(far)}')
        lines.append(f"| {r['layers']}層・{r['heads']}ヘッド | " + ' | '.join(cells) + ' |')
    lines += ['', '## 生成例（評価用の最初の話のタイトルから）', '']
    for r in results:
        g = r['generations'][0]
        lines.append(f"- {r['layers']}層・{r['heads']}ヘッド：{g['text']}（{story.check_story(g['text'])}）")
    lines.append(f"- 正解：{results[0]['generations'][0]['reference']}")
    (ROOT / 'experiments/depth-heads.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
