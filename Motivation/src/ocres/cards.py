"""Cards injected into LLM prompts (hard-rule phrasing for level control).

Decision rule that defines capability level at the parallel-fill state
(pot A cooking AND another pot accepts onions AND Alice is empty-handed):
  L0 (novice)  : MUST pick HOLD  (wait beside the running pot)
  Lk (expert)  : MUST pick FETCH  (go fetch an onion for the other pot)
"""
ALICE_L0 = (
    "经验：你是新手，规则如下——当一口锅已在 cooking 时，你【必须】选 HOLD 在锅旁等待，"
    "绝不去补另一口锅；等它 ready 被伙伴取走、锅空出后再开始新的一锅。"
)
ALICE_LK = (
    "经验：你是熟练厨师，规则如下——当一口锅已在 cooking 且存在另一口空锅或未满的锅(empty/items1/items2)且你空手时，"
    "你【必须】选 FETCH 去拿洋葱，把另一口锅也补上（并行煮两锅），绝不选 HOLD 干等。"
    "只有两锅都 cooking 或都在烹饪/满锅时才 HOLD。锅满3料选 COOK_START；取盘取汤交付由伙伴负责。"
)

BOB_STALE = (
    "你对伙伴 Alice 的印象（可能过时）：她的水平是【新手】——一口锅 cooking 时她会守在旁边 HOLD 等待，"
    "不会去补第二口锅。请以此印象判断她此刻的意图并配合。"
)
BOB_CURRENT = (
    "你对伙伴 Alice 的印象（近期更新）：她的水平是【熟练并行】——一口锅 cooking 且另一口锅能补料时，"
    "她此刻的意图几乎总是 FETCH（去拿洋葱补另一口锅），而不是原地等待。请以此印象判断她此刻的意图并配合。"
)


UPDATED_BASE = (
    "你对伙伴 Alice 的印象（初始估计，待观察更新）：她可能还是新手——一口锅 cooking 时会 HOLD 等待。"
    "但请根据你接下来观察到的她的实际行为持续修正这个印象。"
)


def updated_text(parallel_rate, n):
    if n < 12:
        return UPDATED_BASE
    if parallel_rate >= 0.5:
        return (
            "你对伙伴 Alice 的印象（刚由你的观察更新）：她在最近的行为中频繁在锅 cooking 时去补另一口锅——"
            f"近 {n} 次相关时机里约 {parallel_rate:.0%} 选择补料。你现在判断她是【熟练并行】："
            "她此刻的意图很可能是 FETCH（去拿洋葱补另一口锅），而非 HOLD。"
        )
    return (
        "你对伙伴 Alice 的印象（刚由你的观察更新）：在锅 cooking 的时机里她几乎总是原地等待，"
            f"近 {n} 次相关时机里仅 {parallel_rate:.0%} 选择补料。你判断她仍是【新手串行】：此刻意图多为 HOLD。"
    )
