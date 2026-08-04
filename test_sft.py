import argparse
import json
import re
import sys

import torch
from transformers import AutoTokenizer


def configure_utf8_output():
    """让终端输出使用 UTF-8，避免中文报告出现乱码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


MODEL_DIR = "models/happy-llm-215M-sft"
CHECKPOINT = "checkpoints/sft_report_v5/model.pth"
TOKENIZER_PATH = "tokenizer_k"

sys.path.insert(0, MODEL_DIR)
from k_model import ModelConfig, Transformer


SYSTEM_PROMPT = (
    "你是学习报告助手。程序已经根据原始学习数据计算好报告中的数值。"
    "回答时必须原样保留程序给出的总时长、有效学习时长、有效率、基础分、扣分、最终分和等级，"
    "不要重新计算或修改这些数值。请结合学习模式和分心事件生成完整报告，"
    "报告包括学习结果、行为原因、评分结果、AI分析和AI建议，语气具体、鼓励。"
)


MODE_NAMES = {
    "reading": "阅读学习",
    "courseware": "课件学习",
    "vocabulary": "单词学习",
    "programming": "编程学习",
    "mathematics": "数学学习",
    "online_course": "网课学习",
    "writing": "写作学习",
    "algorithms": "算法学习",
    "late_review": "晚间复习",
    "exam_review": "考试复习",
}


TEST_CASES = [
    {
        "name": "普通阅读",
        "total_duration_sec": 5400,
        "effective_duration_sec": 4680,
        "playing_phone_count": 1,
        "glance_phone_count": 2,
        "away_count": 0,
        "drowsy_count": 0,
        "mode": "reading",
    },
    {
        "name": "高专注课件学习",
        "total_duration_sec": 2400,
        "effective_duration_sec": 2280,
        "playing_phone_count": 0,
        "glance_phone_count": 0,
        "away_count": 0,
        "drowsy_count": 0,
        "mode": "courseware",
    },
    {
        "name": "少量手机干扰的阅读",
        "total_duration_sec": 3600,
        "effective_duration_sec": 3300,
        "playing_phone_count": 0,
        "glance_phone_count": 1,
        "away_count": 0,
        "drowsy_count": 0,
        "mode": "reading",
    },
    {
        "name": "课件学习中的一次离座",
        "total_duration_sec": 7200,
        "effective_duration_sec": 6600,
        "playing_phone_count": 0,
        "glance_phone_count": 0,
        "away_count": 1,
        "drowsy_count": 0,
        "mode": "courseware",
    },
    {
        "name": "课件学习中的多种分心",
        "total_duration_sec": 9000,
        "effective_duration_sec": 5400,
        "playing_phone_count": 2,
        "glance_phone_count": 3,
        "away_count": 1,
        "drowsy_count": 1,
        "mode": "courseware",
    },
    {
        "name": "中等专注阅读",
        "total_duration_sec": 4800,
        "effective_duration_sec": 4080,
        "playing_phone_count": 0,
        "glance_phone_count": 3,
        "away_count": 1,
        "drowsy_count": 0,
        "mode": "reading",
    },
    {
        "name": "无分心课件学习",
        "total_duration_sec": 6000,
        "effective_duration_sec": 5700,
        "playing_phone_count": 0,
        "glance_phone_count": 0,
        "away_count": 0,
        "drowsy_count": 0,
        "mode": "courseware",
    },
    {
        "name": "需要改进的阅读",
        "total_duration_sec": 5400,
        "effective_duration_sec": 4320,
        "playing_phone_count": 1,
        "glance_phone_count": 3,
        "away_count": 1,
        "drowsy_count": 0,
        "mode": "reading",
    },
    {
        "name": "课件学习中的瞌睡",
        "total_duration_sec": 2700,
        "effective_duration_sec": 2400,
        "playing_phone_count": 0,
        "glance_phone_count": 0,
        "away_count": 0,
        "drowsy_count": 1,
        "mode": "courseware",
    },
    {
        "name": "课件学习严重分心并触发下限",
        "total_duration_sec": 7200,
        "effective_duration_sec": 4320,
        "playing_phone_count": 2,
        "glance_phone_count": 4,
        "away_count": 2,
        "drowsy_count": 1,
        "mode": "courseware",
    },
]


def calculate_expected(case):
    base = case["effective_duration_sec"] * 100 // case["total_duration_sec"]
    penalty = (
        case["playing_phone_count"] * 10
        + case["away_count"] * 8
        + case["drowsy_count"] * 15
        + case["glance_phone_count"] * 3
    )
    score = max(0, min(100, base - penalty))

    if score >= 90:
        level = "优秀"
    elif score >= 75:
        level = "良好"
    elif score >= 60:
        level = "一般"
    elif score >= 40:
        level = "需改进"
    else:
        level = "需努力"

    return base, penalty, score, level


def make_event_text(case):
    events = []
    if case["playing_phone_count"]:
        events.append(f"玩手机{case['playing_phone_count']}次")
    if case["glance_phone_count"]:
        events.append(f"看眼手机{case['glance_phone_count']}次")
    if case["away_count"]:
        events.append(f"离座{case['away_count']}次")
    if case["drowsy_count"]:
        events.append(f"低头或瞌睡{case['drowsy_count']}次")
    if not events:
        return "无"
    total = (
        case["playing_phone_count"]
        + case["glance_phone_count"]
        + case["away_count"]
        + case["drowsy_count"]
    )
    return "，".join(events) + f"，共{total}次"


def build_user_input(case, base, penalty, score, level):
    """模拟板端上报原始数据，再由主程序补充计算结果。"""
    total_minutes = case["total_duration_sec"] // 60
    effective_minutes = case["effective_duration_sec"] // 60
    return (
        f"原始学习数据：total_duration_sec={case['total_duration_sec']}；"
        f"effective_duration_sec={case['effective_duration_sec']}；"
        f"playing_phone_count={case['playing_phone_count']}；"
        f"glance_phone_count={case['glance_phone_count']}；"
        f"away_count={case['away_count']}；"
        f"drowsy_count={case['drowsy_count']}；"
        f"mode={case['mode']}。"
        f"程序已计算结果（请勿重新计算）：总时长={total_minutes}分钟；"
        f"有效学习={effective_minutes}分钟；有效率={base}%；基础分={base}；"
        f"扣分={penalty}；最终分={score}；等级={level}。"
    )


def extract_ai_sections(model_text):
    """从模型输出中提取分析和建议，数值字段由程序负责拼接。"""
    analysis_match = re.search(
        r"AI分析\s*[:：]\s*(.*?)(?=AI建议\s*[:：]|$)",
        model_text,
        flags=re.DOTALL,
    )
    advice_match = re.search(
        r"AI建议\s*[:：]\s*(.*)$",
        model_text,
        flags=re.DOTALL,
    )

    analysis = analysis_match.group(1).strip() if analysis_match else ""
    advice = advice_match.group(1).strip() if advice_match else ""

    if not analysis:
        analysis = "本次学习记录已经完成分析，后续可以继续观察专注时间的变化。"
    if not advice:
        advice = "建议根据本次记录调整下一次学习安排，并保持循序渐进。"
    return analysis, advice


def assemble_final_report(case, base, penalty, score, level, model_text):
    """用程序计算的准确字段和模型生成的文字组成最终报告。"""
    total_minutes = case["total_duration_sec"] // 60
    effective_minutes = case["effective_duration_sec"] // 60
    analysis, advice = extract_ai_sections(model_text)
    mode_name = MODE_NAMES.get(case["mode"], case["mode"])
    return (
        f"学习报告；模式：{mode_name}；总时长：{total_minutes}分钟；"
        f"有效学习：{effective_minutes}分钟；有效率：{base}%；"
        f"分心事件：{make_event_text(case)}；评分明细：基础分{base}，扣分{penalty}；"
        f"专注度评分：{score}分，{level}。"
        f"AI分析：{analysis}AI建议：{advice}"
    )


def generate_one(model, tokenizer, device, case, index, temperature=0.6, seed=None):
    base, penalty, score, level = calculate_expected(case)
    user_input = build_user_input(case, base, penalty, score, level)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = tokenizer(prompt).data["input_ids"]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    if seed is not None:
        torch.manual_seed(seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + index)

    with torch.inference_mode():
        output_ids = model.generate(
            x,
            tokenizer.eos_token_id,
            max_new_tokens=256,
            temperature=temperature,
            top_k=300,
        )

    result = tokenizer.decode(
        output_ids[0].tolist(),
        skip_special_tokens=False,
    )

    marker = "<|im_start|>assistant\n"
    if marker in result:
        result = result.split(marker, 1)[1]
    if "<|im_end|>" in result:
        result = result.split("<|im_end|>", 1)[0]

    print(f"\n{'=' * 70}")
    print(f"测试 {index + 1}: {case['name']}")
    print(f"输入：{user_input}")
    print(f"预期：基础分={base}，扣分={penalty}，最终分={score}，等级={level}")
    print(f"事件：{make_event_text(case)}")
    print("模型原始生成：")
    print(result.strip())
    print("程序拼接后的最终报告：")
    print(assemble_final_report(case, base, penalty, score, level, result))


def parse_args():
    parser = argparse.ArgumentParser(
        description="模拟板端学习数据上报、程序计算和 LLM 报告生成流程"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--case",
        type=int,
        help="运行内置测试案例编号，范围为1到10；不指定时运行全部案例",
    )
    source.add_argument(
        "--input_json",
        type=str,
        help="直接传入一条板端原始数据 JSON",
    )
    source.add_argument(
        "--input_file",
        type=str,
        help="从 JSON 文件读取一条板端原始数据",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="生成温度，默认0.6",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="可选的随机种子；不指定时使用真实的随机采样",
    )
    return parser.parse_args()


def load_runtime_case(args):
    if args.input_json:
        case = json.loads(args.input_json)
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            case = json.load(f)
    elif args.case is not None:
        if not 1 <= args.case <= len(TEST_CASES):
            raise ValueError(f"--case 必须在1到{len(TEST_CASES)}之间")
        return [TEST_CASES[args.case - 1]]
    else:
        return TEST_CASES

    required = {
        "total_duration_sec",
        "effective_duration_sec",
        "playing_phone_count",
        "glance_phone_count",
        "away_count",
        "drowsy_count",
        "mode",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"输入数据缺少字段：{sorted(missing)}")
    case.setdefault("name", "实时学习数据")
    return [case]


def main():
    args = parse_args()
    configure_utf8_output()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    config = ModelConfig(
        dim=1024,
        n_layers=18,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=len(tokenizer),
        max_seq_len=512,
        dropout=0.0,
    )

    print("Loading SFT checkpoint...")
    model = Transformer(config)
    state_dict = torch.load(CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    if missing or unexpected:
        raise RuntimeError(
            "SFT checkpoint does not match the inference model architecture. "
            f"Missing keys: {missing}; Unexpected keys: {unexpected}"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print("Device:", device)

    cases = load_runtime_case(args)
    for index, case in enumerate(cases):
        generate_one(
            model,
            tokenizer,
            device,
            case,
            index,
            temperature=args.temperature,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
