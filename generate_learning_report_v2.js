const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const SEED = 20260728;
const TARGET_SIZE = 20000;
const SAMPLE_PATH = path.join(ROOT, "learning_report_sft_v2_20.jsonl");
const OUTPUT_PATH = path.join(ROOT, "learning_report_sft_v2_20000.jsonl");

const SYSTEM_PROMPT =
  "你是学习报告助手。程序已经根据原始学习数据计算好报告中的数值。" +
  "回答时必须原样保留程序给出的总时长、有效学习时长、有效率、基础分、扣分、最终分和等级，" +
  "不要重新计算或修改这些数值。请结合学习模式和分心事件生成完整报告，" +
  "报告包括学习结果、行为原因、评分结果、AI分析和AI建议，语气具体、鼓励。";

const MODES = [
  ["reading", "阅读学习"], ["programming", "编程学习"],
  ["mathematics", "数学学习"], ["vocabulary", "单词学习"],
  ["online_course", "网课学习"], ["writing", "写作学习"],
  ["algorithms", "算法学习"], ["physics", "物理学习"],
  ["exam_review", "考试复习"], ["English", "英语学习"],
  ["chemistry", "化学学习"], ["late_review", "晚间复习"],
  ["project", "项目学习"], ["presentation", "演讲准备"],
  ["code_review", "代码复习"], ["history", "历史学习"],
  ["biology", "生物学习"], ["statistics", "统计学习"],
  ["listening", "听力学习"], ["drawing", "绘图学习"]
];

const ANALYSIS = {
  "优秀": [
    "本次学习有效率达到{base}%，整体过程连续，主要目标完成得比较充分。",
    "本次投入时间大部分转化为了有效学习，专注状态保持得比较稳定。",
    "本次学习节奏清晰，较长时间保持了对当前任务的关注。",
    "本次记录显示学习安排执行得较好，有效时间和总时长之间的差距较小。",
    "本次学习表现出较强的持续投入能力，当前方法对你比较适合。",
    "本次学习完成度较高，分心行为没有明显改变整体的学习效果。"
  ],
  "良好": [
    "本次学习保持了较好的效率，主要学习任务已经完成。",
    "本次投入比较充分，整体节奏稳定，仍有少量细节可以优化。",
    "本次有效学习比例较高，偶尔的中断没有明显影响整体表现。",
    "本次学习已经取得较好效果，连续专注时间基本满足当前任务需要。",
    "本次记录显示学习过程比较稳定，进一步减少短暂打断会带来提升。",
    "本次学习基础较好，时间安排和任务完成情况都处于比较理想的状态。"
  ],
  "一般": [
    "本次有明确的学习投入，但有效时间和总时长之间仍有一定差距。",
    "本次完成了一部分学习目标，连续专注时间还可以继续提高。",
    "本次学习基础尚可，不过几次中断让部分时间没有转化为有效产出。",
    "本次投入并不低，但当前学习节奏还不够稳定，需要减少无关切换。",
    "本次已经形成了一定的学习量，下一步可以把注意力放在提高有效率上。",
    "本次学习有可取之处，分心和疲劳因素仍然影响了任务的连续完成。"
  ],
  "需改进": [
    "本次学习有一定时间投入，但有效率偏低，分心行为对任务连续性影响较明显。",
    "本次完成了部分学习内容，多次中断使有效学习时间明显减少。",
    "本次总时长尚可，但实际专注时间偏少，当前环境需要先做调整。",
    "本次学习存在较多干扰，继续延长时长之前应先恢复稳定的学习节奏。",
    "本次学习目标只完成了一部分，主要问题在于注意力没有持续集中。",
    "本次记录说明投入和产出之间还有差距，减少干扰后效果会更容易提升。"
  ],
  "需努力": [
    "本次总学习时间较长，但有效学习比例偏低，多类分心或疲劳共同影响了效果。",
    "本次学习投入和有效产出差距较大，当前节奏不适合继续单纯增加时长。",
    "本次记录显示连续专注时间不足，学习任务被多次打断。",
    "本次学习状态比较疲惫，分心事件已经明显影响了任务完成情况。",
    "本次有效学习时间有限，先调整环境和作息比立即提高学习时长更重要。",
    "本次结果说明需要重新安排学习区间，但这只是一次状态记录，仍然可以逐步改善。"
  ]
};

const CONTEXT = {
  none: [
    "记录中没有明显分心事件，当前学习环境比较适合继续保持。",
    "整个学习过程中没有出现需要特别处理的行为中断，状态保持得比较完整。",
    "从行为记录看，注意力主要集中在当前任务上，连续学习的基础较好。",
    "本次没有额外的干扰记录，说明你已经找到了一种相对有效的学习安排。"
  ],
  phone: [
    "手机相关行为带来了短暂切换，但整体学习计划仍然完成了一部分。",
    "主动查看或使用手机是本次连续专注被打断的主要原因。",
    "手机分心让部分学习时间没有转化为有效产出，下一次可以优先处理这个因素。",
    "手机相关中断次数较多时，重新进入任务通常需要额外时间，这影响了学习连贯性。"
  ],
  away: [
    "离座事件让当前任务出现了短暂中断，但也可能反映出休息和物品准备还不够规律。",
    "本次有离座行为，提前准备学习用品和饮水可以减少类似打断。",
    "离座对长时间学习的影响需要结合休息安排判断，集中休息通常更容易保持节奏。",
    "当前记录提示学习阶段之间的休息边界还可以安排得更清楚。"
  ],
  drowsy: [
    "低头或瞌睡说明精力在部分时段有所下降，继续硬撑可能降低学习质量。",
    "疲劳状态影响了注意力的持续时间，作息和学习时段值得一起调整。",
    "本次出现了精力不足的信号，缩短学习区间可能比延长总时长更有效。",
    "瞌睡行为不代表能力不足，更像是提醒你需要在精力较好的时间处理重要任务。"
  ],
  mixed: [
    "手机、离座或疲劳行为同时出现，使任务连续性受到多方面影响。",
    "本次分心因素比较集中，先处理最频繁的行为会更容易看到变化。",
    "多种中断叠加后会减少重新进入任务的时间，分阶段调整比一次改变所有习惯更稳妥。",
    "行为记录反映出学习环境和精力管理都有优化空间，但已经有明确的数据可以作为改进起点。"
  ]
};

const ADVICE = {
  phone: [
    "学习前开启专注模式，把手机放到视线外或暂时交给身边的人保管。",
    "把查看消息集中到休息时间，学习阶段只处理当前任务。",
    "开始前关闭无关通知，并先完成一个小目标再接触手机。",
    "可以把手机放到需要起身才能拿到的位置，降低随手查看的可能性。",
    "为手机设置明确的查看时段，减少短暂查看带来的连续打断。"
  ],
  glance: [
    "把想查看的事情先记下来，等一个学习阶段结束后统一处理。",
    "可以使用定时专注区间，专注期间暂不查看手机。",
    "把手机屏幕朝下并移出视线，降低无意识查看的频率。",
    "下次先安排一段不查看手机的连续学习时间，再根据完成情况调整。",
    "如果只是担心错过消息，可以提前设置紧急联系人，其他通知暂时静音。"
  ],
  away: [
    "学习前准备好草稿纸、饮水和资料，把必要的离座安排在阶段之间。",
    "将较长任务拆成几个阶段，在阶段结束时集中休息和整理物品。",
    "提前检查桌面和设备，减少为了寻找物品而离开座位。",
    "给自己设置固定的休息节点，避免在任务中途频繁离座。",
    "休息是学习的一部分，可以有计划地安排，而不是等到任务被打断后才离开。"
  ],
  drowsy: [
    "优先保证睡眠、照明和坐姿，出现疲劳时及时休息。",
    "把重要内容安排到精力较好的时段，晚间复习可以缩短单次区间。",
    "可以采用二十五到四十分钟的学习区间，区间结束后主动恢复精力。",
    "先调整作息和学习环境，再逐步增加学习时间，不必一次追求很长的时长。",
    "出现困倦时先做简单整理或短暂休息，恢复后再处理需要集中思考的内容。"
  ],
  general: [
    "继续保持当前节奏，并在学习结束后回顾本次完成的重点内容。",
    "把下一次任务拆成一个个可完成的小目标，每完成一个就记录进展。",
    "开始前写下最重要的一项任务，结束时检查它是否已经完成。",
    "保留本次有效的学习安排，同时只调整一个最明显的干扰因素。",
    "可以记录连续专注的时间，看到小幅进步后再逐步增加任务难度。"
  ]
};

const SECOND_ADVICE = [
  "每次只做一个小调整，连续坚持几天后再根据记录决定下一步。",
  "即使一次只多保持十分钟，也能为后续学习建立更稳定的节奏。",
  "完成阶段目标后给自己一个短暂的正向反馈，有助于保持动力。",
  "下次结束时记录一个做得好的地方和一个可改进的地方，进步会更容易被看见。",
  "如果当天状态较差，可以降低任务量，但尽量保留短时间的连续学习。",
  "你已经有了可观察的学习记录，按照数据逐步调整比追求一次改变所有问题更可靠。",
  "保持规律比偶尔长时间学习更重要，稳定的积累会带来更好的结果。",
  "建议先完成最重要的任务，再处理次要内容，减少开始阶段的犹豫。"
];

function makeRng(seed) {
  let value = seed >>> 0;
  return function random() {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick(rng, values) {
  return values[Math.floor(rng() * values.length)];
}

function intRange(rng, low, high) {
  return low + Math.floor(rng() * (high - low + 1));
}

function weightedCategory(rng) {
  const x = rng();
  if (x < 0.18) return "优秀";
  if (x < 0.43) return "良好";
  if (x < 0.75) return "一般";
  if (x < 0.90) return "需改进";
  return "需努力";
}

function getLevel(score) {
  if (score >= 90) return "优秀";
  if (score >= 75) return "良好";
  if (score >= 60) return "一般";
  if (score >= 40) return "需改进";
  return "需努力";
}

function chooseEvents(rng, category) {
  let playing, glance, away, drowsy;
  if (category === "优秀") {
    playing = rng() < 0.04 ? 1 : 0;
    glance = intRange(rng, 0, 1);
    away = rng() < 0.08 ? 1 : 0;
    drowsy = 0;
  } else if (category === "良好") {
    playing = rng() < 0.18 ? 1 : 0;
    glance = intRange(rng, 0, 2);
    away = rng() < 0.22 ? 1 : 0;
    drowsy = rng() < 0.05 ? 1 : 0;
  } else if (category === "一般") {
    playing = intRange(rng, 0, 2);
    glance = intRange(rng, 1, 4);
    away = intRange(rng, 0, 2);
    drowsy = rng() < 0.15 ? 1 : 0;
  } else if (category === "需改进") {
    playing = intRange(rng, 1, 3);
    glance = intRange(rng, 1, 5);
    away = intRange(rng, 1, 3);
    drowsy = rng() < 0.35 ? intRange(rng, 1, 2) : 0;
  } else {
    playing = intRange(rng, 2, 5);
    glance = intRange(rng, 2, 8);
    away = intRange(rng, 1, 4);
    drowsy = intRange(rng, 1, 3);
  }
  return {playing, glance, away, drowsy};
}

function eventText(events) {
  const parts = [];
  if (events.playing) parts.push(`玩手机${events.playing}次`);
  if (events.glance) parts.push(`看眼手机${events.glance}次`);
  if (events.away) parts.push(`离座${events.away}次`);
  if (events.drowsy) parts.push(`低头或瞌睡${events.drowsy}次`);
  if (!parts.length) return "无";
  const total = events.playing + events.glance + events.away + events.drowsy;
  return parts.join("，") + `，共${total}次`;
}

function contextKey(events) {
  const types = [events.playing > 0, events.glance > 0, events.away > 0, events.drowsy > 0].filter(Boolean).length;
  if (types === 0) return "none";
  if (types > 1) return "mixed";
  if (events.drowsy) return "drowsy";
  if (events.away) return "away";
  return "phone";
}

function adviceKey(events) {
  if (events.drowsy) return "drowsy";
  if (events.playing) return "phone";
  if (events.away) return "away";
  if (events.glance) return "glance";
  return "general";
}

function makeCase(rng, index) {
  const category = weightedCategory(rng);
  const totalMinutes = intRange(rng, 2, 16) * 15 + 15;
  let ratioLow, ratioHigh;
  if (category === "优秀") { ratioLow = 0.94; ratioHigh = 1.00; }
  else if (category === "良好") { ratioLow = 0.86; ratioHigh = 0.94; }
  else if (category === "一般") { ratioLow = 0.74; ratioHigh = 0.88; }
  else if (category === "需改进") { ratioLow = 0.62; ratioHigh = 0.82; }
  else { ratioLow = 0.45; ratioHigh = 0.70; }

  const ratio = ratioLow + rng() * (ratioHigh - ratioLow);
  const effectiveMinutes = Math.max(1, Math.min(totalMinutes, Math.round(totalMinutes * ratio)));
  const events = chooseEvents(rng, category);
  const totalSeconds = totalMinutes * 60;
  const effectiveSeconds = effectiveMinutes * 60;
  const base = Math.floor(effectiveSeconds * 100 / totalSeconds);
  const penalty = events.playing * 10 + events.away * 8 + events.drowsy * 15 + events.glance * 3;
  const score = Math.max(0, Math.min(100, base - penalty));
  const grade = getLevel(score);
  const mode = pick(rng, MODES);
  const analysisLead = pick(rng, ANALYSIS[grade]).replace("{base}", String(base));
  const analysis = analysisLead + pick(rng, CONTEXT[contextKey(events)]);
  const advice = pick(rng, ADVICE[adviceKey(events)]) + pick(rng, SECOND_ADVICE);

  const user =
    `原始学习数据：total_duration_sec=${totalSeconds}；effective_duration_sec=${effectiveSeconds}；` +
    `playing_phone_count=${events.playing}；glance_phone_count=${events.glance}；` +
    `away_count=${events.away}；drowsy_count=${events.drowsy}；mode=${mode[0]}。` +
    `程序已计算结果（请勿重新计算）：总时长=${totalMinutes}分钟；有效学习=${effectiveMinutes}分钟；` +
    `有效率=${base}%；基础分=${base}；扣分=${penalty}；最终分=${score}；等级=${grade}。`;
  const assistant =
    `学习报告；模式：${mode[1]}；总时长：${totalMinutes}分钟；有效学习：${effectiveMinutes}分钟；` +
    `有效率：${base}%；分心事件：${eventText(events)}；评分明细：基础分${base}，扣分${penalty}；` +
    `专注度评分：${score}分，${grade}。AI分析：${analysis}AI建议：${advice}`;
  return JSON.stringify([
    {role: "system", content: SYSTEM_PROMPT},
    {role: "user", content: user},
    {role: "assistant", content: assistant}
  ]);
}

const rng = makeRng(SEED);
const sampleLines = fs.readFileSync(SAMPLE_PATH, "utf8").split(/\r?\n/).filter(Boolean);
if (sampleLines.length !== 20) throw new Error(`样例数据应为20条，实际为${sampleLines.length}条`);
const lines = sampleLines.slice();
for (let i = lines.length; i < TARGET_SIZE; i++) lines.push(makeCase(rng, i));
fs.writeFileSync(OUTPUT_PATH, lines.join("\n") + "\n", "utf8");
console.log(`Generated ${lines.length} records: ${OUTPUT_PATH}`);
