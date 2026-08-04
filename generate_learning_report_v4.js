const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const size = Number(process.argv[2] || 20000);
const outputName = process.argv[3] || `learning_report_sft_v4_${size}.jsonl`;
const outputPath = path.join(ROOT, outputName);
const SEED = 20260730;

const SYSTEM_PROMPT =
  "你是学习报告助手。程序已经根据原始学习数据计算好报告中的数值。" +
  "回答时必须原样保留程序给出的总时长、有效学习时长、有效率、基础分、扣分、最终分和等级，" +
  "不要重新计算或修改这些数值。请结合学习模式和分心事件生成完整报告，" +
  "报告包括学习结果、行为原因、评分结果、AI分析和AI建议，语气具体、鼓励。";

const MODES = [
  ["reading", "看书学习", "章节、观点和重点内容", "阅读任务"],
  ["courseware", "课件学习", "课程小节、讲义和课堂重点", "课件学习任务"]
].map(([key, name, focus, task]) => ({key, name, focus, task}));

const GRADE_START = {
  "优秀": [
    "本次有效率为{base}%，有{effective}分钟都投入到了学习中，整体节奏保持得比较稳定。",
    "从{total}分钟的记录看，本次学习的有效比例达到{base}%，主要目标完成情况较好。",
    "本次专注度评分为{score}分，说明大部分学习时间都转化成了有效投入。",
    "本次学习的投入和产出比较接近，有效率{base}%体现出较好的时间利用情况。",
    "这次记录显示学习安排执行得比较完整，有效时长达到{effective}分钟。"
  ],
  "良好": [
    "本次有效率为{base}%，主要任务已经完成，整体学习节奏比较稳定。",
    "本次获得了{score}分，说明投入时间大部分转化成了有效学习。",
    "在{total}分钟的学习中，有效时长为{effective}分钟，当前状态总体良好。",
    "本次学习已经取得较好效果，仍有少量时间可以进一步转化为有效产出。",
    "从记录看，本次学习基础较好，连续投入的时间足以支撑当前任务。"
  ],
  "一般": [
    "本次有效率为{base}%，已经完成了一部分学习目标，但仍有时间没有转化为有效投入。",
    "本次专注度为{score}分，说明学习有明确投入，连续性还可以继续加强。",
    "在{total}分钟的记录中，有效学习为{effective}分钟，当前节奏还有调整空间。",
    "本次学习基础尚可，分心行为让部分计划内容没有按预期完成。",
    "这次记录体现出一定的学习量，下一步可以把重点放在提高有效率上。"
  ],
  "需改进": [
    "本次有效率为{base}%，有效学习时间偏少，多次中断影响了任务连续性。",
    "本次得到{score}分，说明已经有学习投入，但当前环境还不利于保持专注。",
    "在{total}分钟的学习中，只有{effective}分钟转化为有效学习，需要先调整节奏。",
    "本次完成了部分内容，剩余时间受到行为中断和注意力变化的影响。",
    "这次记录暴露出一些可以改善的地方，减少干扰后有效时间会更容易增加。"
  ],
  "需努力": [
    "本次有效率只有{base}%，{total}分钟的学习中有效时长为{effective}分钟，连续投入明显不足。",
    "本次专注度为{score}分，多类行为中断已经明显影响了任务完成情况。",
    "这次学习的投入和产出差距较大，先恢复稳定节奏比继续延长时长更重要。",
    "本次只完成了部分学习内容，当前状态需要从环境和精力管理开始调整。",
    "这是一条需要重视的状态记录，但分数反映的是本次表现，不代表你的学习能力。"
  ]
};

const MODE_ANALYSIS = [
  "在{focus}的学习中，保留前后内容之间的联系有助于下一次继续推进；本次有效率为{base}%。",
  "本次{task}涉及{focus}，把阶段性成果记录下来可以减少重新进入任务的时间；本次持续了{total}分钟。",
  "对于{task}来说，按小目标推进{focus}会比一次处理所有内容更容易保持节奏；本次得分为{score}分。",
  "{focus}需要一定的连续思考，本次记录可以作为下一次安排学习区间的参考；有效学习时长为{effective}分钟。",
  "如果能在每个阶段结束时回顾{focus}，本次投入会更容易转化为长期积累；本次记录包含{eventTotal}次分心事件。"
];

const EVENT_CONTEXT = {
  none: [
    "本次没有记录到明显分心行为，当前环境与学习安排比较匹配，有效率为{base}%。",
    "行为记录保持干净，注意力主要集中在当前任务上，本次有效学习达到{effective}分钟。",
    "没有额外中断影响本次过程，可以继续观察这种安排是否适合长期保持，下一次可安排{next}分钟的连续区间。",
    "本次学习没有出现突出的干扰信号，连续完成任务的基础比较好，当前评分为{score}分。"
  ],
  phone: [
    "记录到{events}，手机切换让当前任务中断了{eventTotal}次。",
    "本次主要干扰来自{events}，重新回到{task}时会消耗额外注意力。",
    "手机相关行为占据了部分学习间隙，减少{events}会更容易提升有效率。",
    "虽然每次查看时间可能不长，但{events}的累积会影响{focus}的连续理解。"
  ],
  away: [
    "本次出现{events}，提前准备物品可以减少对当前任务的打断。",
    "{events}让学习过程出现了短暂空档，集中安排休息会更利于保持节奏。",
    "离座次数为{eventTotal}次，说明休息和物品准备仍有进一步规划的空间。",
    "本次中断主要与离座有关，阶段之间安排固定休息可以减少临时离开。"
  ],
  drowsy: [
    "本次记录到{events}，精力变化影响了后半段的注意力持续时间。",
    "{events}说明当前学习时段可能不在精力高峰，继续延长时间未必有效。",
    "疲劳行为共出现{eventTotal}次，缩短学习区间并及时恢复精力会更稳妥。",
    "本次的主要限制来自精力状态，先保证恢复再处理需要集中思考的内容。"
  ],
  mixed: [
    "本次同时出现{events}，多种中断叠加后会增加重新进入任务的成本。",
    "行为记录包括{events}，先处理最频繁的因素会更容易看到变化。",
    "多类干扰共同影响了{focus}的连续推进，分阶段调整比一次改变所有习惯更稳妥。",
    "本次的中断因素比较集中，记录中的数量可以帮助确定下一次的优先改进项。"
  ]
};

const EVENT_ADVICE = {
  phone: [
    "开始前可以开启专注模式，把手机放到视线之外，先完成{next}分钟的任务。",
    "把查看消息集中到休息时间，学习阶段先处理当前任务，减少第{eventTotal}次切换。",
    "提前关闭无关通知，完成一个小目标后再接触手机，本次目标可以设为{next}分钟。",
    "如果担心漏掉重要消息，可以只保留紧急联系人提醒，让{task}保持连续。",
    "把手机放到需要起身才能拿到的位置，降低随手查看的机会，先保持{next}分钟专注。"
  ],
  glance: [
    "想查看的事情可以先记在纸上，完成{next}分钟的学习阶段后统一处理。",
    "设置一段不查看手机的专注区间，再根据{task}的完成情况安排休息。",
    "将手机屏幕朝下并移出视线，先把{focus}连续处理{next}分钟。",
    "把短暂查看合并到固定休息点，避免在{task}中频繁切换注意力。",
    "下次先完成一个连续学习目标，再检查第{eventTotal}次查看时想处理的消息。"
  ],
  away: [
    "学习前准备好资料、饮水和用品，把必要的离座安排在{next}分钟阶段之间。",
    "开始前检查桌面和设备，减少为了寻找物品而中途离开{task}。",
    "把较长任务拆成几个阶段，在第{eventTotal}次离座前后集中休息和整理。",
    "给自己设置固定的休息节点，避免{focus}进行到一半才临时离座。",
    "合理休息有助于持续学习，可以把{task}的休息从临时中断变成计划安排。"
  ],
  drowsy: [
    "优先调整睡眠、照明和坐姿，出现疲劳时及时休息，下一段可先安排{next}分钟。",
    "把重要的{focus}放到精力较好的时段，当前任务可以缩短单次区间。",
    "先恢复精力再继续学习，不必为了延长总时长而强行坚持{task}。",
    "可以使用{next}分钟左右的学习区间，结束后主动恢复精力。",
    "观察第{eventTotal}次疲劳出现的时间，把需要集中思考的任务提前安排。"
  ],
  general: [
    "继续保持当前节奏，并在结束后回顾本次完成的{focus}。",
    "把下一次{task}拆成几个小目标，每完成一个就记录进展。",
    "开始前写下最重要的一项{task}，结束时检查完成情况。",
    "保留本次有效的安排，只调整一个最明显的干扰因素，先尝试{next}分钟。",
    "记录连续专注时间，看到小幅进步后再逐步增加{focus}的难度。"
  ]
};

const SUPPORTIVE = [
  "每次只做一个小调整，连续坚持{days}天后再根据记录决定下一步；本次得分为{score}分。",
  "即使一次只多保持{next}分钟，也能为后续学习建立更稳定的节奏；本次有效率为{base}%。",
  "完成阶段目标后给自己一个短暂的正向反馈，有助于保持{task}的动力；本次有效学习为{effective}分钟。",
  "下次结束时记录一个做得好的地方和一个可改进的地方，进步会更容易被看见；本次持续了{total}分钟。",
  "保持规律比偶尔长时间学习更重要，稳定积累{effective}分钟的有效时间会带来更好的结果；本次评分为{score}分。",
  "先完成最重要的{focus}，再处理次要内容，开始时会更容易进入状态；下一段可安排{next}分钟。",
  "如果当天状态较差，可以降低{task}的任务量，但尽量保留一段连续学习时间；本次有效率为{base}%。",
  "把本次的有效做法留下来，下一次只需要继续完善{focus}中的一个细节；本次得分为{score}分。"
];

// 报告中的建议不重复复述有效率、时长和分数，只描述下一步行动。
const SIMPLE_MODE_ADVICE = {
  reading: [
    "阅读前先确定这一轮要找的一个问题，结束时用自己的话复述。",
    "把章节中的重点句做标记，读完一小段再整理，减少来回切换。",
    "遇到需要查证的内容先记下来，完成当前阅读后统一处理。",
    "可以先确定一个清晰的小目标，完成后再决定是否继续深入。",
    "把前后段落之间的联系写在页边，下一次更容易接着读。",
    "阅读结束前回看标记内容，确认哪些观点已经真正理解。",
    "把较难的章节拆开处理，先完成能够独立理解的部分。",
    "读完后用几句话概括内容，帮助知识从短时记忆留下来。",
    "把阅读和整理分成两个阶段，减少在不同任务之间来回切换。",
    "选择一个安静的开始动作，例如先打开材料并写下阅读目标。",
    "下一次可以沿用本次有效的环境，只调整一个明显的干扰来源。",
    "完成一个段落后短暂回顾，比一次处理大量内容更容易保持专注。"
  ],
  courseware: [
    "每看完一个小节就记下关键词，便于之后快速回顾。",
    "遇到听不懂的部分先暂停标记，完成当前小节后再集中回放。",
    "课件开始前准备好讲义和笔记位置，减少中途寻找资料。",
    "把课程重点分成几个小块，每完成一块就检查是否理解。",
    "看完课程后用自己的话复述主要内容，不要只依赖再次播放。",
    "把需要练习的知识点单独列出来，之后安排针对性复习。",
    "尽量在一个小节结束后再处理其他消息，保持内容的连贯性。",
    "对照讲义记录疑问，集中整理比频繁跳出课程更高效。",
    "课程较长时可以分段完成，给每一段留下清晰的笔记。",
    "先明确本次要掌握的一个重点，听课时更容易筛选信息。",
    "把重要概念和例子放在一起记录，复习时更容易建立联系。",
    "结束后保留一个待复习清单，下一次从最薄弱的部分开始。"
  ]
};

const SIMPLE_EVENT_ADVICE = {
  phone: [
    "学习开始前把手机放到视线外，降低顺手拿起的机会。",
    "把需要处理的消息留到休息阶段，学习时只保留必要提醒。",
    "如果有重要消息，可以提前设置联系人提醒，减少主动查看。",
    "把手机放到需要起身才能拿到的位置，帮助当前任务保持连续。",
    "先完成当前学习片段，再统一处理手机上的事务。"
  ],
  glance: [
    "想查看的事情先记在纸上，结束当前学习片段后统一处理。",
    "将手机屏幕朝下并移出视线，给正在处理的内容留出连续时间。",
    "把短暂查看合并到休息阶段，避免在任务中频繁切换注意力。",
    "开始学习前先清理无关通知，减少注意力被突然拉走的机会。",
    "如果想到其他事情，先写下关键词，稍后再集中处理。"
  ],
  away: [
    "学习前准备好资料、饮水和用品，减少为了取东西而中途离开。",
    "开始前检查桌面和设备，把必要的休息安排在任务阶段之间。",
    "给自己设置固定的休息节点，避免任务进行到一半才临时离座。",
    "把较长任务拆成几个阶段，在阶段之间集中整理和走动。",
    "合理休息有助于持续学习，让离座从临时中断变成计划安排。"
  ],
  drowsy: [
    "优先调整睡眠、照明和坐姿，出现疲劳时及时恢复精力。",
    "把需要集中思考的内容放到精力较好的时段处理。",
    "先恢复状态再继续学习，不必为了延长时间而勉强坚持。",
    "观察疲劳通常在什么情况下出现，下一次提前调整学习安排。",
    "如果已经明显困倦，可以先休息，再处理需要理解和记忆的内容。"
  ],
  general: [
    "继续保持当前节奏，并在结束后回顾本次完成的内容。",
    "把下一次任务拆成几个小目标，每完成一个就记录进展。",
    "开始前写下最重要的一项任务，结束时检查完成情况。",
    "保留本次有效的安排，只调整一个最明显的干扰因素。",
    "记录连续专注时的感受，逐步找到适合自己的学习节奏。"
  ]
};

const SIMPLE_SUPPORTIVE = [
  "保持规律比偶尔长时间学习更重要，稳定积累会带来变化。",
  "给自己一个短暂的正向反馈，下一次会更容易重新开始。",
  "先从最容易做到的一步开始，不必一次改变所有习惯。",
  "把本次有效的做法保留下来，之后继续观察它是否适合自己。",
  "如果当天状态不同，可以根据实际精力灵活安排任务。",
  "稳定完成眼前的小目标，长期效果会比追求一次做很多更好。",
  "为学习留出明确的开始动作，进入状态会更加顺利。",
  "结束后记下一点收获和一点困难，下一次调整会更有依据。",
  "把注意力放回当前任务，完成后再处理其他事情。",
  "循序渐进地调整环境和习惯，持续行动比短暂冲刺更可靠。"
];

const ADVICE_SUFFIXES = [
  "接下来可以继续围绕{focus}推进。",
  "这项调整也适合放在{task}开始前执行。",
  "先从最容易做到的一步开始。",
  "把本次有效做法保留下来，下一次继续观察。",
  "完成后记下一点感受，便于之后调整。",
  "给自己留出明确的开始动作，进入状态会更容易。",
  "如果当天状态不同，可以按实际精力灵活安排。",
  "保持温和而稳定的节奏，逐步形成自己的方法。",
  "本次{modeName}可以沿用这个思路，再做小幅调整。",
  "围绕{task}保持连续推进，比反复切换更有效。",
  "把这项建议写进学习计划，执行起来会更清楚。",
  "先处理最重要的部分，剩余内容可以分阶段完成。"
];

function activeModeAdvice(mode) {
  return SIMPLE_MODE_ADVICE[mode.key] || SIMPLE_MODE_ADVICE.reading;
}

function activeEventAdvice(events) {
  return SIMPLE_EVENT_ADVICE[adviceKey(events)] || SIMPLE_EVENT_ADVICE.general;
}

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

function fill(template, values) {
  return template.replace(/\{(\w+)\}/g, (_, key) => String(values[key]));
}

function getLevel(score) {
  if (score >= 90) return "优秀";
  if (score >= 75) return "良好";
  if (score >= 60) return "一般";
  if (score >= 40) return "需改进";
  return "需努力";
}

function chooseCategory(rng, index) {
  if (index < 20) return ["优秀", "良好", "一般", "需改进", "需努力"][index % 5];
  const value = rng();
  if (value < 0.18) return "优秀";
  if (value < 0.43) return "良好";
  if (value < 0.75) return "一般";
  if (value < 0.90) return "需改进";
  return "需努力";
}

function chooseEvents(rng, category) {
  if (category === "优秀") return {playing: rng() < 0.04 ? 1 : 0, glance: intRange(rng, 0, 1), away: rng() < 0.08 ? 1 : 0, drowsy: 0};
  if (category === "良好") return {playing: rng() < 0.18 ? 1 : 0, glance: intRange(rng, 0, 2), away: rng() < 0.22 ? 1 : 0, drowsy: rng() < 0.05 ? 1 : 0};
  if (category === "一般") return {playing: intRange(rng, 0, 2), glance: intRange(rng, 1, 4), away: intRange(rng, 0, 2), drowsy: rng() < 0.15 ? 1 : 0};
  if (category === "需改进") return {playing: intRange(rng, 1, 3), glance: intRange(rng, 1, 5), away: intRange(rng, 1, 3), drowsy: rng() < 0.35 ? intRange(rng, 1, 2) : 0};
  return {playing: intRange(rng, 2, 5), glance: intRange(rng, 2, 8), away: intRange(rng, 1, 4), drowsy: intRange(rng, 1, 3)};
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

function eventKey(events) {
  const types = [events.playing, events.glance, events.away, events.drowsy].filter(Boolean).length;
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

// 只有在候选句与已有句子重复时才追加这些自然的上下文，避免训练语料出现完全相同的句子。
// 如果上下文也发生碰撞，最后的序号只作为兜底，不会改变报告中的计算结果。
const UNIQUE_SUFFIXES = [
  "本次{modeName}记录的有效率为{base}%，得分为{score}分。",
  "这段安排共持续{total}分钟，其中有效学习为{effective}分钟。",
  "记录中的分心事件共{eventTotal}次，可以作为下一次调整的参照。",
  "下一段可以先安排{next}分钟，连续观察{days}天再复盘。",
  "本次保留下来的有效学习时长是{effective}分钟。",
  "把这次{score}分的记录和下一次表现放在一起比较，会更容易看到变化。",
  "当前{modeName}任务的总时长为{total}分钟，建议按阶段继续观察。",
  "这次记录的基础分为{base}%，下一次可以从一个小目标开始。",
  "本次记录中的事件是{events}，先处理最明显的一项即可。",
  "针对{focus}，可以把下一段学习设置为{next}分钟左右。",
  "对于{task}，本次有效学习为{effective}分钟，值得继续保持。",
  "本次{modeName}的有效率为{base}%，对应评分为{score}分。",
  "把{events}作为下一次复盘时的观察重点，调整一个因素就够了。",
  "本次总时长为{total}分钟，连续坚持{days}天后再比较变化。",
  "下一次继续处理{focus}时，可以先回顾本次留下的进度。",
  "这次{task}的记录显示有效学习达到{effective}分钟。"
];

function makeUniqueSentence(template, values, seen, local, recordIndex, slot, suffixes = UNIQUE_SUFFIXES) {
  // 每个候选保持为一个完整句子，避免把重复的前半句和不同的后半句拼成两个句子。
  const base = fill(template, values).replace(/。+$/, "");
  const candidates = [`${base}。`];
  for (let i = 0; i < suffixes.length; i++) {
    const suffix = fill(suffixes[(i + recordIndex + slot) % suffixes.length], values).replace(/。+$/, "");
    candidates.push(`${base}；${suffix}。`);
  }
  for (const candidate of candidates) {
    if (!seen.has(candidate) && !local.has(candidate)) {
      local.add(candidate);
      return candidate;
    }
  }
  const fallback = suffixes === ADVICE_SUFFIXES
    ? `${base}，可以作为下一次学习的一个小调整。`
    : `${base}，本次可作为第${recordIndex + 1}条记录的第${slot + 1}个观察点。`;
  local.add(fallback);
  return fallback;
}

function makeCase(rng, index, seen) {
  const category = chooseCategory(rng, index);
  // 使用更细的时长范围，避免大量样本共享相同的句子参数。
  const totalMinutes = intRange(rng, 30, 240);
  const ranges = {"优秀": [0.94, 1.00], "良好": [0.86, 0.94], "一般": [0.74, 0.88], "需改进": [0.62, 0.82], "需努力": [0.45, 0.70]};
  const [low, high] = ranges[category];
  const ratio = low + rng() * (high - low);
  const effectiveMinutes = Math.max(1, Math.min(totalMinutes, Math.round(totalMinutes * ratio)));
  const events = chooseEvents(rng, category);
  const totalSeconds = totalMinutes * 60;
  const effectiveSeconds = effectiveMinutes * 60;
  const base = Math.floor(effectiveSeconds * 100 / totalSeconds);
  const penalty = events.playing * 10 + events.away * 8 + events.drowsy * 15 + events.glance * 3;
  const score = Math.max(0, Math.min(100, base - penalty));
  const grade = getLevel(score);
  const mode = index < MODES.length ? MODES[index] : pick(rng, MODES);
  const eventTotal = events.playing + events.glance + events.away + events.drowsy;
  const eventDescription = eventText(events);
  const next = intRange(rng, 20, 50);
  const days = intRange(rng, 3, 10);
  const values = {
    base,
    score,
    total: totalMinutes,
    effective: effectiveMinutes,
    focus: mode.focus,
    task: mode.task,
    modeName: mode.name,
    events: eventDescription,
    eventTotal,
    next,
    days
  };
  const local = new Set();
  const analysis = [
    makeUniqueSentence(pick(rng, GRADE_START[grade]), values, seen, local, index, 0),
    makeUniqueSentence(pick(rng, MODE_ANALYSIS), values, seen, local, index, 1),
    makeUniqueSentence(pick(rng, EVENT_CONTEXT[eventKey(events)]), values, seen, local, index, 2)
  ];
  const advice = [
    makeUniqueSentence(pick(rng, activeModeAdvice(mode)), values, seen, local, index, 3, ADVICE_SUFFIXES),
    makeUniqueSentence(pick(rng, activeEventAdvice(events)), values, seen, local, index, 4, ADVICE_SUFFIXES),
    makeUniqueSentence(pick(rng, SIMPLE_SUPPORTIVE), values, seen, local, index, 5, ADVICE_SUFFIXES)
  ];
  const sentences = analysis.concat(advice);
  sentences.forEach(sentence => seen.add(sentence));
  const user =
    `原始学习数据：total_duration_sec=${totalSeconds}；effective_duration_sec=${effectiveSeconds}；` +
    `playing_phone_count=${events.playing}；glance_phone_count=${events.glance}；` +
    `away_count=${events.away}；drowsy_count=${events.drowsy}；mode=${mode.key}。` +
    `程序已计算结果（请勿重新计算）：总时长=${totalMinutes}分钟；有效学习=${effectiveMinutes}分钟；` +
    `有效率=${base}%；基础分=${base}；扣分=${penalty}；最终分=${score}；等级=${grade}。`;
  const assistant =
    `学习报告；模式：${mode.name}；总时长：${totalMinutes}分钟；有效学习：${effectiveMinutes}分钟；` +
    `有效率：${base}%；分心事件：${eventDescription}；评分明细：基础分${base}，扣分${penalty}；` +
    `专注度评分：${score}分，${grade}。AI分析：${analysis.join("")}AI建议：${advice.join("")}`;
  return JSON.stringify([{role: "system", content: SYSTEM_PROMPT}, {role: "user", content: user}, {role: "assistant", content: assistant}]);
}

function modeAdvice(mode) {
  const advice = {
    reading: ["下次可按章节拆分{focus}，完成一段后写下两条要点。", "阅读{focus}时先确定一个问题，结束后用自己的话复述答案。", "把{focus}中需要查证的部分标记下来，连续阅读完成后再统一处理。"],
    programming: ["可以先写下{focus}的下一步任务，减少重新进入代码时的思考成本。", "处理{focus}中的报错时先记录现象和尝试，集中排查比频繁切换更有效。", "每完成一段{focus}就保存进展和说明，帮助下一次继续编程任务。"],
    mathematics: ["下次可以围绕{focus}安排一组小题，完成后再集中整理错题。", "遇到{focus}中的难题先标记卡住的位置，保持解题流程，休息时再回顾。", "练习结束后写出一条关于{focus}的解题思路，比单纯增加题量更有帮助。"],
    vocabulary: ["可以把{focus}放进短句中复习，并在结束时再次快速回顾。", "将{focus}分成小组，下一次先复习旧词再加入新内容。", "保持短时间重复练习，让{focus}逐步进入长期记忆。"],
    online_course: ["把{focus}拆成小节，每完成一节就暂停整理三条重点。", "观看{focus}前准备好讲义，遇到难点先标记，休息时再集中回看。", "网课开始前关闭无关通知，尽量在{focus}的小节结束后再处理其他事务。"],
    writing: ["先确定{focus}中的一个段落目标，再集中处理措辞和格式。", "写作阶段先记录{focus}的想法，不必频繁修改已经完成的内容。", "每完成一段{focus}就保存进展，休息后可以更快回到原来的思路。"],
    algorithms: ["先写出{focus}的步骤，再用一个小例子检查每一步。", "把{focus}的复杂度和适用条件记在例题旁边，方便比较不同方法。", "完成一类{focus}后整理共同思路，逐步形成自己的解题模板。"],
    physics: ["解答{focus}时先列出已知条件和待求量，再选择对应公式。", "把{focus}中容易混淆的物理量单独整理，最后检查单位和边界条件。", "遇到{focus}中的难题先记录卡住的位置，休息后再从条件分析开始。"],
    exam_review: ["先列出{focus}中最需要巩固的知识点，再安排短时段练习。", "每个复习区间结束后记录{focus}中仍不熟悉的内容，避免无目的重复。", "把{focus}中的模拟题和错题分开安排，逐步提高复习的针对性。"],
    English: ["把{focus}中的新表达写进自己的句子，并在结束时朗读一遍。", "可以把{focus}分成小段，完成一段后复述主要意思。", "保持规律练习，定期回顾{focus}中掌握不牢的表达。"],
    chemistry: ["把{focus}的反应条件和现象整理成对照表，优先复习易混部分。", "学习{focus}后配一道例题，检查是否真正理解了应用条件。", "将{focus}拆成小组，每次复习一组并及时回顾。"],
    late_review: ["把{focus}安排成较短区间，结束后及时休息，不必勉强延长。", "优先回顾{focus}中的重要内容，剩余任务留到精力更好的时段。", "睡前用几分钟整理{focus}即可，保证睡眠比继续低效学习更重要。"],
    project: ["把{focus}拆成可独立完成的小模块，每次只处理当前模块。", "结束前记录{focus}的进度和下一步计划，下一次可以更快恢复。", "遇到{focus}中的阻塞时先写下问题，再安排集中处理时间。"],
    presentation: ["先完成一轮{focus}的完整演练，再针对一个细节进行修改。", "记录{focus}中最需要改进的一个细节，避免同时调整太多内容。", "练习结束后回看{focus}的表现，选择一个具体问题作为下一轮目标。"],
    code_review: ["先确定{focus}中需要检查的模块，按文件或功能分段完成。", "把{focus}中发现的问题写在清单中，集中处理比反复跳转更高效。", "每完成一个{focus}模块就记录结论，下一次从未完成部分继续。"],
    history: ["按时间线整理{focus}的主要事件，并补充关键原因或影响。", "把{focus}中容易混淆的人物和年代列出，复习结束后快速自测。", "先完成一个历史阶段的{focus}梳理，再处理跨阶段比较。"],
    biology: ["用图示标出{focus}之间的关系，再用自己的话复述过程。", "把{focus}中的相似概念放在一起比较，找出共同点和区别。", "完成一个{focus}主题后做几道小题，检查记忆是否转化为理解。"],
    statistics: ["先说明{focus}中指标代表什么，再进行公式计算。", "每学习一个{focus}方法就配一个小数据例子，检查结果是否合理。", "把{focus}中不同方法的使用条件整理成表格，优先复习易混部分。"],
    listening: ["把{focus}分成短段，先完整听一遍，再针对遗漏部分精听。", "听完{focus}后用自己的话复述主要内容，再查看原文核对。", "选择长度合适的{focus}材料规律练习，逐步增加难度。"],
    drawing: ["先确定{focus}中的构图或局部目标，再集中处理细节。", "每完成一个{focus}阶段就保存作品，记录一个需要改进的地方。", "把{focus}的观察、起稿和修改分成阶段，减少频繁切换。"]
  };
  return advice[mode.key];
}

function pickSupportive(rng) {
  return pick(rng, SUPPORTIVE);
}

const rng = makeRng(SEED);
const seen = new Set();
const lines = [];
for (let i = 0; i < size; i++) lines.push(makeCase(rng, i, seen));
fs.writeFileSync(outputPath, lines.join("\n") + "\n", "utf8");
console.log(`Generated ${lines.length} records: ${outputPath}`);
console.log(`Unique analysis/advice sentences: ${seen.size}`);
