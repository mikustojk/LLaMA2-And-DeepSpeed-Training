import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "learning_report_sft_100.jsonl"
OUTPUT_PATH = ROOT / "learning_report_sft_10000.jsonl"
TARGET_SIZE = 10000
SEED = 20260728

SYSTEM_PROMPT = (
    "你是学习报告助手。按规则计算：基础分=有效学习时长×100÷总学习时长并向下取整；"
    "扣分=玩手机次数×10+离座次数×8+低头或瞌睡次数×15+看眼手机次数×3；"
    "最终分限制在0到100。等级为90到100优秀、75到89良好、60到74一般、"
    "40到59需改进、低于40需努力。报告按学习结果、行为原因、评分结果、AI建议组织，语气鼓励。"
)

MODES = [
    ("reading", "阅读学习"),
    ("programming", "编程学习"),
    ("mathematics", "数学学习"),
    ("vocabulary", "单词学习"),
    ("online_course", "网课学习"),
    ("writing", "写作学习"),
    ("algorithms", "算法学习"),
    ("physics", "物理学习"),
    ("exam_review", "考试复习"),
    ("English", "英语学习"),
    ("chemistry", "化学学习"),
    ("late_review", "晚间复习"),
    ("project", "项目学习"),
    ("presentation", "演讲准备"),
    ("code_review", "代码复习"),
    ("history", "历史学习"),
    ("biology", "生物学习"),
    ("statistics", "统计学习"),
    ("listening", "听力学习"),
    ("drawing", "绘图学习"),
]

ANALYSIS = {
    "优秀": [
        "本次有效学习比例很高，学习过程连续，整体状态非常稳定。",
        "本次学习目标完成充分，有效学习时间和专注度都表现良好。",
        "本次学习保持了较高效率，记录到的分心事件对整体过程影响很小。",
        "本次学习节奏清晰，投入时间能够较好地转化为有效学习。",
    ],
    "良好": [
        "本次有效学习比例较高，整体学习过程稳定，仍有少量可优化的中断。",
        "本次学习已经取得较好效果，个别分心事件影响了部分连续专注时间。",
        "本次学习完成度较好，专注状态总体稳定，可以继续优化学习环境。",
        "本次学习投入比较充分，少量行为中断没有改变整体的良好表现。",
    ],
    "一般": [
        "本次已经完成了一定的有效学习，但分心事件影响了连续专注，仍有提升空间。",
        "本次学习有明确投入，不过有效率和任务连续性还可以进一步提高。",
        "本次学习完成了一部分目标，主要问题集中在中途打断或疲劳状态。",
        "本次学习基础较好，但部分时间没有转化为有效学习，需要减少无关中断。",
    ],
    "需改进": [
        "本次学习投入有一定基础，但有效率偏低，分心事件对学习节奏影响较明显。",
        "本次学习完成了部分任务，不过多次中断使有效学习时间明显减少。",
        "本次总时长尚可，实际专注时间偏少，下一次需要先改善学习环境。",
        "本次学习过程中存在较多干扰，先缩短单次学习时长会更容易恢复节奏。",
    ],
    "需努力": [
        "本次总学习时间较长，但有效学习比例偏低，多类分心事件共同影响了学习效果。",
        "本次学习投入和有效产出差距较大，频繁中断或疲劳状态是主要影响因素。",
        "本次记录显示连续专注时间不足，先建立短时间、低干扰的学习区间更合适。",
        "本次学习需要从减少干扰和调整节奏开始，逐步恢复稳定的有效学习时间。",
    ],
}

PHONE_ADVICE = [
    "学习前把手机放到视线外或开启专注模式，完成一个学习阶段后再统一查看消息。",
    "将手机放到不易拿到的位置，规定只在休息时间查看，减少主动切换任务。",
    "开始学习前关闭无关通知，先完成一个明确的小目标，再处理手机上的其他事务。",
]

GLANCE_ADVICE = [
    "把查看手机集中到休息时间，保持学习阶段内的连续专注。",
    "学习期间先记录想查看的事情，等阶段结束后统一处理，减少短暂打断。",
    "可以使用定时专注区间，专注期间不查看手机，休息时再集中处理消息。",
]

AWAY_ADVICE = [
    "提前准备学习用品和饮水，并安排固定休息时间，减少非计划离座。",
    "把必要的休息集中到学习阶段之间，减少离座对当前任务的打断。",
    "开始学习前整理好桌面和所需物品，再按阶段安排休息，保持学习连续性。",
]

DROWSY_ADVICE = [
    "优先保证休息和学习环境的舒适度，缩短单次学习时长，避免疲劳状态下继续学习。",
    "可以把学习安排到精力较好的时段，使用较短的学习区间，并在出现疲劳时及时休息。",
    "先调整睡眠、照明和坐姿，再逐步恢复学习时间，不建议在明显困倦时强行延长学习。",
]

GENERAL_ADVICE = [
    "继续保持当前节奏，并在学习结束后简要回顾本次内容。",
    "把学习目标拆成较小阶段，每个阶段结束后检查完成情况。",
    "在开始前写下最重要的一项任务，结束时核对目标是否完成。",
    "记录本次学习中掌握不牢的内容，作为下一次学习的起点。",
]


def get_level(score: int) -> str:
    if score >= 90:
        return "优秀"
    if score >= 75:
        return "良好"
    if score >= 60:
        return "一般"
    if score >= 40:
        return "需改进"
    return "需努力"


def event_text(playing: int, glance: int, away: int, drowsy: int) -> str:
    parts = []
    if playing:
        parts.append(f"玩手机{playing}次")
    if glance:
        parts.append(f"看眼手机{glance}次")
    if away:
        parts.append(f"离座{away}次")
    if drowsy:
        parts.append(f"低头或瞌睡{drowsy}次")
    total = playing + glance + away + drowsy
    if not parts:
        return "无"
    return "，".join(parts) + f"，共{total}次"


def make_case(rng: random.Random, index: int) -> list[dict[str, str]]:
    # 用不同状态区间生成数据，保证训练集中包含五个评分等级。
    category = rng.choices(
        ["优秀", "良好", "一般", "需改进", "需努力"],
        weights=[18, 25, 32, 15, 10],
        k=1,
    )[0]

    total_minutes = rng.choice(range(30, 241, 15))
    if category == "优秀":
        ratio = rng.uniform(0.94, 1.00)
        playing = rng.choices([0, 1], weights=[95, 5], k=1)[0]
        glance = rng.randint(0, 1)
        away = rng.choices([0, 1], weights=[90, 10], k=1)[0]
        drowsy = 0
    elif category == "良好":
        ratio = rng.uniform(0.86, 0.94)
        playing = rng.choices([0, 1], weights=[80, 20], k=1)[0]
        glance = rng.randint(0, 2)
        away = rng.choices([0, 1], weights=[75, 25], k=1)[0]
        drowsy = rng.choices([0, 1], weights=[95, 5], k=1)[0]
    elif category == "一般":
        ratio = rng.uniform(0.74, 0.88)
        playing = rng.randint(0, 2)
        glance = rng.randint(1, 4)
        away = rng.randint(0, 2)
        drowsy = rng.choices([0, 1], weights=[85, 15], k=1)[0]
    elif category == "需改进":
        ratio = rng.uniform(0.62, 0.82)
        playing = rng.randint(1, 3)
        glance = rng.randint(1, 5)
        away = rng.randint(1, 3)
        drowsy = rng.choices([0, 1, 2], weights=[60, 30, 10], k=1)[0]
    else:
        ratio = rng.uniform(0.45, 0.70)
        playing = rng.randint(2, 5)
        glance = rng.randint(2, 8)
        away = rng.randint(1, 4)
        drowsy = rng.randint(1, 3)

    effective_minutes = max(1, min(total_minutes, round(total_minutes * ratio)))
    total_seconds = total_minutes * 60
    effective_seconds = effective_minutes * 60
    base = effective_seconds * 100 // total_seconds
    penalty = playing * 10 + away * 8 + drowsy * 15 + glance * 3
    score = max(0, min(100, base - penalty))
    grade = get_level(score)
    mode, mode_name = MODES[index % len(MODES)]

    if drowsy:
        advice = rng.choice(DROWSY_ADVICE)
    elif playing:
        advice = rng.choice(PHONE_ADVICE)
    elif away:
        advice = rng.choice(AWAY_ADVICE)
    elif glance:
        advice = rng.choice(GLANCE_ADVICE)
    else:
        advice = rng.choice(GENERAL_ADVICE)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"学习数据：total_duration_sec={total_seconds}；"
                f"effective_duration_sec={effective_seconds}；"
                f"playing_phone_count={playing}；glance_phone_count={glance}；"
                f"away_count={away}；drowsy_count={drowsy}；mode={mode}。"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"学习报告；模式：{mode_name}；总时长：{total_minutes}分钟；"
                f"有效学习：{effective_minutes}分钟；有效率：{base}%；"
                f"分心事件：{event_text(playing, glance, away, drowsy)}；"
                f"评分明细：基础分{base}，扣分{penalty}；"
                f"专注度评分：{score}分，{grade}。"
                f"AI分析：{rng.choice(ANALYSIS[grade])}"
                f"AI建议：{advice}"
            ),
        },
    ]
    return messages


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"找不到种子数据：{SOURCE_PATH}")

    seed_lines = SOURCE_PATH.read_text(encoding="utf-8").splitlines()
    if len(seed_lines) != 100:
        raise ValueError(f"种子数据应为100条，实际为{len(seed_lines)}条")
    for line in seed_lines:
        json.loads(line)

    rng = random.Random(SEED)
    lines = list(seed_lines)
    while len(lines) < TARGET_SIZE:
        lines.append(json.dumps(make_case(rng, len(lines)), ensure_ascii=False))

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {len(lines)} 条数据：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
