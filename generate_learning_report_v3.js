const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const size = Number(process.argv[2] || 20000);
const outputName = process.argv[3] || `learning_report_sft_v3_${size}.jsonl`;
const outputPath = path.join(ROOT, outputName);
const SEED = 20260729;

const SYSTEM_PROMPT =
  "你是学习报告助手。程序已经根据原始学习数据计算好报告中的数值。" +
  "回答时必须原样保留程序给出的总时长、有效学习时长、有效率、基础分、扣分、最终分和等级，" +
  "不要重新计算或修改这些数值。请结合学习模式和分心事件生成完整报告，" +
  "报告包括学习结果、行为原因、评分结果、AI分析和AI建议，语气具体、鼓励。";

const MODES = [
  {key: "reading", name: "阅读学习", analysis: [
    "阅读内容已经完成了一部分，可以继续关注章节之间的理解和回顾。",
    "本次阅读的连续性会直接影响对重点内容的记忆，分段整理有助于保持思路。"
  ], advice: [
    "下次可以按章节或小节设置阅读目标，并在阶段结束时写下两三条要点。",
    "阅读前先确定本次要解决的问题，结束后用自己的话复述重点内容。",
    "把需要查证的内容先标记下来，连续阅读完成后再集中处理。"
  ]},
  {key: "programming", name: "编程学习", analysis: [
    "编程任务需要连续保留上下文，短暂切换也可能增加重新理解代码的时间。",
    "本次编程学习已经形成了一定进展，记录问题和解决步骤有助于后续衔接。"
  ], advice: [
    "可以先写下当前模块的下一步任务，减少重新打开代码时的思考成本。",
    "遇到报错时先记录现象和尝试过的方法，集中排查比频繁切换任务更有效。",
    "每完成一个功能模块就保存进展和简短说明，帮助维持编程学习的连续性。"
  ]},
  {key: "mathematics", name: "数学学习", analysis: [
    "数学题目的思路需要连续推导，稳定的学习区间有助于保留前面的解题过程。",
    "本次数学学习已经完成了一定练习量，错题整理可以帮助巩固本次收获。"
  ], advice: [
    "下次可以按题型安排小目标，完成一组题后再集中整理错题。",
    "把暂时不会的题目先标记下来，保持当前解题流程，休息时再回顾。",
    "练习结束后写出一条解题思路，比单纯增加题目数量更有助于巩固。"
  ]},
  {key: "vocabulary", name: "单词学习", analysis: [
    "单词学习适合短时重复和间隔复习，当前的连续投入有利于形成记忆积累。",
    "本次词汇学习已经完成了较明确的内容，容易混淆的词语值得单独记录。"
  ], advice: [
    "可以把新词放进短句中复习，并在当天结束时再次快速回顾。",
    "把易混淆词分成小组，下一次先复习旧词再加入新词。",
    "保持每天短时间练习，稳定重复比一次记忆大量单词更容易坚持。"
  ]},
  {key: "online_course", name: "网课学习", analysis: [
    "网课内容通常按小节推进，保持观看和记录的连续性有助于跟上课程结构。",
    "本次网课学习已经完成了一部分，暂停整理重点可以减少只观看不吸收的情况。"
  ], advice: [
    "把网课拆成小节，每完成一节就暂停整理三条重点。",
    "观看前准备好讲义，遇到难点先标记，休息时再集中回看。",
    "网课开始前关闭无关通知，尽量在每个小节结束后再处理其他事务。"
  ]},
  {key: "writing", name: "写作学习", analysis: [
    "写作需要连续保持主题和结构，减少中途切换有助于保留表达思路。",
    "本次写作已经形成了一定产出，下一步可以关注提纲和段落之间的衔接。"
  ], advice: [
    "先确定本次要完成的段落或提纲，再集中处理措辞和格式。",
    "写作阶段先记录想法，不必频繁修改已经写好的内容。",
    "每完成一个小段落就保存进展，休息后可以更快回到原来的思路。"
  ]},
  {key: "algorithms", name: "算法学习", analysis: [
    "算法学习需要同时理解步骤和复杂度，连续练习有助于建立完整的推导过程。",
    "本次算法学习已经积累了一定内容，整理关键步骤可以加深对方法的理解。"
  ], advice: [
    "下一次可以先写出算法步骤，再用一个小例子验证每一步。",
    "把复杂度和适用条件记录在例题旁边，复习时会更容易比较不同方法。",
    "完成一类题目后整理共同思路，逐步形成自己的解题模板。"
  ]},
  {key: "physics", name: "物理学习", analysis: [
    "物理题目需要结合公式、条件和图示，连续思考有助于保持完整的分析链条。",
    "本次物理学习已经完成了部分练习，归纳公式使用条件可以巩固理解。"
  ], advice: [
    "下次解题时先写出已知条件和待求量，再选择对应公式。",
    "把容易混淆的物理量单独整理，完成练习后再检查单位和边界条件。",
    "遇到不会的题目先记录卡住的位置，休息后再从条件分析开始尝试。"
  ]},
  {key: "exam_review", name: "考试复习", analysis: [
    "考试复习需要兼顾知识回顾和题目练习，合理分段比单纯延长时长更重要。",
    "本次复习已经覆盖了一部分内容，下一步可以根据薄弱点调整复习顺序。"
  ], advice: [
    "先列出最需要巩固的知识点，再安排短时段练习和回顾。",
    "每个复习区间结束后记录仍不熟悉的内容，避免无目的重复。",
    "把模拟题和错题分开安排，逐步提高复习的针对性。"
  ]},
  {key: "English", name: "英语学习", analysis: [
    "英语学习适合听、读、写交替练习，保持稳定频率有利于形成长期积累。",
    "本次英语学习已经完成了明确内容，结合例句使用可以进一步巩固记忆。"
  ], advice: [
    "把本次遇到的新表达写进自己的句子中，结束后再快速朗读一遍。",
    "可以把听力或阅读材料分成小段，完成一段后复述主要意思。",
    "保持每天短时间练习，并定期回顾之前掌握不牢的表达。"
  ]},
  {key: "chemistry", name: "化学学习", analysis: [
    "化学学习需要同时记忆概念和理解反应条件，分段整理有助于保持条理。",
    "本次化学学习已经完成了一部分知识梳理，重点反应和易错条件值得继续巩固。"
  ], advice: [
    "把反应条件和现象整理成对照表，复习时先回顾容易混淆的部分。",
    "完成知识点学习后配合一道例题，检查是否真正理解了应用条件。",
    "将需要记忆的内容拆成小组，每次复习一组并及时回顾。"
  ]},
  {key: "late_review", name: "晚间复习", analysis: [
    "晚间复习更需要关注精力变化，合理控制区间有助于保留学习效果。",
    "本次晚间复习已经完成了一定内容，疲劳信号提示作息安排还可以继续优化。"
  ], advice: [
    "把晚间复习安排成较短区间，结束后及时休息，不要勉强延长时间。",
    "优先回顾当天最重要的内容，剩余任务留到精力更好的时段处理。",
    "睡前用几分钟整理重点即可，保证睡眠比继续低效学习更重要。"
  ]},
  {key: "project", name: "项目学习", analysis: [
    "项目任务通常包含多个环节，持续记录进度有助于保持整体目标和当前步骤的联系。",
    "本次项目学习已经推进了一部分，明确下一步交付内容可以减少重新规划的时间。"
  ], advice: [
    "把项目拆成可以独立完成的小模块，每次只处理当前模块。",
    "结束前记录已完成内容和下一步计划，下一次可以更快恢复工作状态。",
    "遇到阻塞时先写下问题和需要查找的资料，再安排集中处理时间。"
  ]},
  {key: "presentation", name: "演讲准备", analysis: [
    "演讲准备需要反复练习和及时复盘，连续完成一个段落有助于提高表达稳定性。",
    "本次演讲准备已经取得一定进展，重点内容和表达节奏还可以继续打磨。"
  ], advice: [
    "先完成一轮完整演练，再针对停顿、语速或重点表达进行修改。",
    "把每次练习中最需要改进的一个细节记录下来，避免同时调整太多内容。",
    "练习结束后回听或回看一遍，选择一个具体问题作为下一轮目标。"
  ]},
  {key: "code_review", name: "代码复习", analysis: [
    "代码复习需要持续保留文件结构和修改背景，频繁切换会增加重新理解的成本。",
    "本次代码复习已经检查了一部分内容，记录修改原因有助于后续继续排查。"
  ], advice: [
    "先确定需要检查的模块，按文件或功能分段完成复习。",
    "把发现的问题和修改建议写在清单中，集中处理比反复跳转更高效。",
    "每完成一个模块就记录结论，下一次可以直接从未完成部分继续。"
  ]},
  {key: "history", name: "历史学习", analysis: [
    "历史学习需要建立时间线和事件之间的联系，连续整理有助于形成整体理解。",
    "本次历史学习已经积累了一定材料，比较不同事件的原因和影响可以加深记忆。"
  ], advice: [
    "按时间线整理主要事件，并为每个事件补充一个关键原因或影响。",
    "把容易混淆的人物和年代单独列出，复习结束后快速自测。",
    "先完成一个历史阶段的梳理，再处理跨阶段的比较问题。"
  ]},
  {key: "biology", name: "生物学习", analysis: [
    "生物学习需要把概念、结构和过程联系起来，分层整理能减少零散记忆。",
    "本次生物学习已经完成了部分知识点，图示和关键词复习可以帮助巩固。"
  ], advice: [
    "用图示标出结构之间的关系，再用自己的话复述过程。",
    "把相似概念放在一起比较，找出它们的共同点和区别。",
    "每完成一个主题就做几道小题，检查记忆是否能够转化为理解。"
  ]},
  {key: "statistics", name: "统计学习", analysis: [
    "统计学习需要同时理解指标含义和计算步骤，连续练习有助于减少概念混淆。",
    "本次统计学习已经完成了一部分内容，结合数据案例可以进一步确认理解是否准确。"
  ], advice: [
    "先说明指标代表什么，再进行公式计算，避免只记步骤而忽略含义。",
    "每学习一个方法就配一个小数据例子，完成后检查结果是否符合直觉。",
    "把不同统计方法的使用条件整理成表格，复习时优先查看易混部分。"
  ]},
  {key: "listening", name: "听力学习", analysis: [
    "听力学习需要连续接收信息，保持短时专注比反复中断更有利于理解上下文。",
    "本次听力练习已经完成了一定内容，复述和精听可以帮助确认真正听懂的部分。"
  ], advice: [
    "把材料分成短段，先完整听一遍，再针对遗漏部分进行精听。",
    "听完后用自己的话复述主要内容，最后再查看原文核对。",
    "选择长度合适的材料保持规律练习，逐步增加难度而不是一次延长时长。"
  ]},
  {key: "drawing", name: "绘图学习", analysis: [
    "绘图学习需要连续保持观察和操作，稳定的练习区间有助于保留创作思路。",
    "本次绘图已经完成了一定练习，记录构图和细节问题可以帮助下一次继续改进。"
  ], advice: [
    "先确定构图或局部练习目标，再集中处理细节。",
    "每完成一个阶段就保存作品并记录一个需要改进的地方。",
    "把观察、起稿和修改分成不同阶段，减少频繁切换带来的干扰。"
  ]}
];

const GRADE_ANALYSIS = {
  "优秀": [
    "本次有效率达到{base}%，整体学习过程连续，主要目标完成得比较充分。",
    "本次投入时间大部分转化为了有效学习，专注状态保持得比较稳定。",
    "本次学习节奏清晰，较长时间保持了对当前任务的关注。"
  ],
  "良好": [
    "本次学习保持了较好的效率，主要学习任务已经完成。",
    "本次投入比较充分，整体节奏稳定，仍有少量细节可以优化。",
    "本次有效学习比例较高，偶尔的中断没有明显影响整体表现。"
  ],
  "一般": [
    "本次有明确的学习投入，但有效时间和总时长之间仍有一定差距。",
    "本次完成了一部分学习目标，连续专注时间还可以继续提高。",
    "本次学习基础尚可，不过几次中断让部分时间没有转化为有效产出。"
  ],
  "需改进": [
    "本次学习有一定时间投入，但有效率偏低，分心行为对任务连续性影响较明显。",
    "本次完成了部分学习内容，多次中断使有效学习时间明显减少。",
    "本次总时长尚可，但实际专注时间偏少，当前环境需要先做调整。"
  ],
  "需努力": [
    "本次总学习时间较长，但有效学习比例偏低，多类分心或疲劳共同影响了效果。",
    "本次学习投入和有效产出差距较大，当前节奏不适合继续单纯增加时长。",
    "本次记录显示连续专注时间不足，学习任务被多次打断。"
  ]
};

const EVENT_CONTEXT = {
  none: [
    "记录中没有明显分心事件，当前学习环境比较适合继续保持。",
    "整个学习过程中没有出现需要特别处理的行为中断，状态保持得比较完整。"
  ],
  phone: [
    "手机相关行为带来了短暂切换，减少这类中断后更容易保持连续学习。",
    "主动查看或使用手机是本次专注被打断的主要原因，可以优先调整这个因素。"
  ],
  away: [
    "离座行为让当前任务出现了短暂中断，提前准备物品可以减少类似打断。",
    "本次离座次数有限，但把休息安排在阶段之间会更有利于保持节奏。"
  ],
  drowsy: [
    "低头或瞌睡说明精力在部分时段有所下降，作息和学习时段值得一起调整。",
    "疲劳状态影响了注意力的持续时间，缩短单次区间可能比延长总时长更有效。"
  ],
  mixed: [
    "手机、离座或疲劳行为同时出现，使任务连续性受到多方面影响。",
    "多种中断叠加后会减少重新进入任务的时间，分阶段调整会更稳妥。"
  ]
};

const EVENT_ADVICE = {
  phone: [
    "学习前开启专注模式，把手机放到视线外。",
    "把查看消息集中到休息时间，学习阶段只处理当前任务。",
    "开始前关闭无关通知，并先完成一个小目标再接触手机。"
  ],
  glance: [
    "把想查看的事情先记下来，等学习阶段结束后统一处理。",
    "可以使用定时专注区间，专注期间暂不查看手机。",
    "把手机移出视线，降低无意识查看的频率。"
  ],
  away: [
    "学习前准备好草稿纸、饮水和资料，把必要的离座安排在阶段之间。",
    "将较长任务拆成几个阶段，在阶段结束时集中休息和整理物品。",
    "提前检查桌面和设备，减少为了寻找物品而离开座位。"
  ],
  drowsy: [
    "优先保证睡眠、照明和坐姿，出现疲劳时及时休息。",
    "把重要内容安排到精力较好的时段，缩短单次学习区间。",
    "先调整作息和学习环境，再逐步增加学习时间。"
  ],
  general: [
    "继续保持当前节奏，并在学习结束后回顾本次完成的重点内容。",
    "把下一次任务拆成一个个可完成的小目标，每完成一个就记录进展。",
    "开始前写下最重要的一项任务，结束时检查它是否已经完成。"
  ]
};

const SUPPORTIVE_SENTENCES = [
  "每次只做一个小调整，连续坚持几天后再根据记录决定下一步。",
  "即使一次只多保持十分钟，也能为后续学习建立更稳定的节奏。",
  "完成阶段目标后给自己一个短暂的正向反馈，有助于保持动力。",
  "下次结束时记录一个做得好的地方和一个可改进的地方，进步会更容易被看见。",
  "保持规律比偶尔长时间学习更重要，稳定的积累会带来更好的结果。"
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
  if (category === "优秀") {
    return {
      playing: rng() < 0.04 ? 1 : 0,
      glance: intRange(rng, 0, 1),
      away: rng() < 0.08 ? 1 : 0,
      drowsy: 0
    };
  }
  if (category === "良好") {
    return {
      playing: rng() < 0.18 ? 1 : 0,
      glance: intRange(rng, 0, 2),
      away: rng() < 0.22 ? 1 : 0,
      drowsy: rng() < 0.05 ? 1 : 0
    };
  }
  if (category === "一般") {
    return {
      playing: intRange(rng, 0, 2),
      glance: intRange(rng, 1, 4),
      away: intRange(rng, 0, 2),
      drowsy: rng() < 0.15 ? 1 : 0
    };
  }
  if (category === "需改进") {
    return {
      playing: intRange(rng, 1, 3),
      glance: intRange(rng, 1, 5),
      away: intRange(rng, 1, 3),
      drowsy: rng() < 0.35 ? intRange(rng, 1, 2) : 0
    };
  }
  return {
    playing: intRange(rng, 2, 5),
    glance: intRange(rng, 2, 8),
    away: intRange(rng, 1, 4),
    drowsy: intRange(rng, 1, 3)
  };
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

function makeCase(rng, index) {
  const category = chooseCategory(rng, index);
  const totalMinutes = intRange(rng, 2, 16) * 15 + 15;
  const ranges = {
    "优秀": [0.94, 1.00], "良好": [0.86, 0.94], "一般": [0.74, 0.88],
    "需改进": [0.62, 0.82], "需努力": [0.45, 0.70]
  };
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
  const lead = pick(rng, GRADE_ANALYSIS[grade]).replace("{base}", String(base));
  const analysis = lead + pick(rng, mode.analysis) + pick(rng, EVENT_CONTEXT[eventKey(events)]);
  const advice = pick(rng, mode.advice) + pick(rng, EVENT_ADVICE[adviceKey(events)]) + pick(rng, SUPPORTIVE_SENTENCES);
  const user =
    `原始学习数据：total_duration_sec=${totalSeconds}；effective_duration_sec=${effectiveSeconds}；` +
    `playing_phone_count=${events.playing}；glance_phone_count=${events.glance}；` +
    `away_count=${events.away}；drowsy_count=${events.drowsy}；mode=${mode.key}。` +
    `程序已计算结果（请勿重新计算）：总时长=${totalMinutes}分钟；有效学习=${effectiveMinutes}分钟；` +
    `有效率=${base}%；基础分=${base}；扣分=${penalty}；最终分=${score}；等级=${grade}。`;
  const assistant =
    `学习报告；模式：${mode.name}；总时长：${totalMinutes}分钟；有效学习：${effectiveMinutes}分钟；` +
    `有效率：${base}%；分心事件：${eventText(events)}；评分明细：基础分${base}，扣分${penalty}；` +
    `专注度评分：${score}分，${grade}。AI分析：${analysis}AI建议：${advice}`;
  return JSON.stringify([
    {role: "system", content: SYSTEM_PROMPT},
    {role: "user", content: user},
    {role: "assistant", content: assistant}
  ]);
}

const rng = makeRng(SEED);
const lines = [];
for (let i = 0; i < size; i++) lines.push(makeCase(rng, i));
fs.writeFileSync(outputPath, lines.join("\n") + "\n", "utf8");
console.log(`Generated ${lines.length} records: ${outputPath}`);
