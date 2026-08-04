const fs = require("fs");
const path = require("path");

const SOURCE_PATH = "learning_report_sft_v4_source_10000.jsonl";
const SEED_PATH = "learning_report_sft_v5_20_revised.jsonl";
const OUTPUT_PATH = "data/sft/learning_report_sft_v5_10000.jsonl";
const SIZE = 10000;
const SEED = 20260731;

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

function numberFrom(text, pattern) {
  const match = text.match(pattern);
  if (!match) throw new Error(`无法从输入中读取字段：${pattern}`);
  return Number(match[1]);
}

function eventDescription(events) {
  const parts = [];
  if (events.playing) parts.push(`玩手机${events.playing}次`);
  if (events.glance) parts.push(`看眼手机${events.glance}次`);
  if (events.away) parts.push(`离座${events.away}次`);
  if (events.drowsy) parts.push(`低头或瞌睡${events.drowsy}次`);
  return parts.length ? parts.join("、") : "无";
}

function parseUser(user) {
  const events = {
    playing: numberFrom(user, /playing_phone_count=([0-9]+)/),
    glance: numberFrom(user, /glance_phone_count=([0-9]+)/),
    away: numberFrom(user, /away_count=([0-9]+)/),
    drowsy: numberFrom(user, /drowsy_count=([0-9]+)/)
  };
  const base = numberFrom(user, /有效率=([0-9]+)%/);
  const score = numberFrom(user, /最终分=([0-9]+)/);
  const grade = (user.match(/等级=([^。]+)。/) || [])[1];
  const total = events.playing + events.glance + events.away + events.drowsy;
  return {
    base,
    score,
    grade,
    events,
    eventTotal: total,
    eventText: eventDescription(events)
  };
}

const EFFICIENCY_HIGH = [
  "本次学习有效率达到{base}%且时间利用情况比较理想",
  "本次学习有效率为{base}%且大部分时间都投入到了有效学习中",
  "从本次记录看有效率{base}%说明整体投入比较充分",
  "本次有效率为{base}%且学习时间的利用情况保持得不错",
  "本次学习有{base}%的时间处于有效状态且整体表现比较稳定",
  "本次记录显示有效率达到{base}%且专注时间占比较高"
];

const EFFICIENCY_MIDDLE = [
  "本次学习有效率为{base}%且还有一部分时间没有转化为有效投入",
  "本次有效率达到{base}%说明整体基础不错但仍有提升空间",
  "从{base}%的有效率来看本次学习已经有一定投入",
  "本次学习有效率为{base}%且时间利用情况处在可以继续改善的水平",
  "本次有{base}%的时间用于有效学习且当前节奏还需要进一步稳定",
  "有效率为{base}%说明学习过程有收获但连续性还可以加强"
];

const EFFICIENCY_LOW = [
  "本次学习有效率为{base}%且有较多时间没有转化为有效投入",
  "本次有效率只有{base}%说明当前学习时间的利用还不够充分",
  "从{base}%的有效率来看本次学习过程受到了一定影响",
  "本次学习有效率为{base}%且有效投入的比例还有明显提升空间",
  "本次只有{base}%的时间处于有效学习状态需要先调整学习节奏",
  "有效率为{base}%说明本次投入不够连续后续可以逐步改善"
];

const DISTRACTION_NONE = [
  "本次没有记录到分心行为且注意力保持得比较集中",
  "学习过程中没有出现明显中断且当前状态比较稳定",
  "本次分心次数为0且连续投入的基础比较好",
  "本次记录没有出现需要特别关注的干扰行为",
  "学习期间没有分心事件且整体过程保持得比较平稳",
  "本次注意力主要集中在当前安排上且分心影响很小"
];

const DISTRACTION_PREFIX = [
  "本次共记录到{count}次分心行为并表现为{events}",
  "学习过程中出现了{count}次分心并包括{events}",
  "本次有{count}次行为中断并记录到{events}",
  "分心行为共出现{count}次并主要涉及{events}",
  "从行为记录看本次发生了{count}次分心并表现为{events}",
  "本次学习受到{count}次分心影响并包括{events}"
];

const DISTRACTION_SUFFIX = [
  "这些中断会影响注意力的连续性",
  "多次切换会增加重新进入状态的难度",
  "减少其中较频繁的行为有助于保持节奏",
  "这说明当前环境还有可以调整的地方",
  "适当减少中断后有效投入会更容易增加",
  "这些行为共同影响了学习过程的稳定性"
];

const CONCLUSION = {
  "优秀": [
    "整体学习状态比较稳定并可继续保持当前节奏",
    "当前表现已经比较理想且保持规律会更有帮助",
    "本次基础状态很好且后续重点是维持这种连续性"
  ],
  "良好": [
    "整体表现较好且减少零散干扰后还会更加稳定",
    "当前学习基础不错且稍微调整环境就能继续提升",
    "本次状态比较积极并保持规律减少中断即可"
  ],
  "一般": [
    "如果减少这些干扰有效学习时间还有提升空间",
    "当前投入已经有一定基础且下一步可以先改善连续性",
    "只要逐步减少中断学习效果会比本次更加稳定"
  ],
  "需改进": [
    "当前状态需要先减少干扰再逐步恢复稳定投入",
    "如果把最明显的中断控制住有效率会更容易提高",
    "下一步可以从改善环境和恢复节奏开始调整"
  ],
  "需努力": [
    "当前学习过程受到较多影响且先恢复规律比延长时间更重要",
    "减少主要干扰并合理安排休息后状态才更容易恢复",
    "本次表现需要调整但通过小步改变仍然可以逐渐改善"
  ]
};

const ANALYSIS_DETAILS = [
  "继续保持规律比偶尔长时间坚持更重要",
  "把当前状态延续下去会更容易看到变化",
  "逐步调整比一次改变所有习惯更稳妥",
  "目前最重要的是先找到适合自己的节奏",
  "稳定的投入会比短暂的冲刺更有帮助",
  "只要减少主要干扰状态就会逐渐改善",
  "给自己留出恢复和调整的空间更有利于坚持",
  "保持耐心并记住短期波动不代表长期表现",
  "当前记录可以作为下一次调整的参考",
  "从一个小变化开始会更容易执行",
  "有意识地观察状态有助于找到问题来源",
  "让安排保持清楚并使执行起来更加轻松",
  "把有效的做法保留下来并在后续继续验证",
  "逐步建立稳定习惯比追求一次完美更实际",
  "关注连续性会比单看某一次结果更有帮助",
  "只要愿意持续调整表现就有机会提升",
  "把注意力放回当前事情并让学习体验更平稳",
  "不要因为一次状态不佳否定自己的努力",
  "合理安排休息也属于学习计划的一部分",
  "坚持记录之后会更容易发现自己的变化"
];

const ADVICE_CLAUSE_1 = [
  "学习开始前先整理好周围环境，减少无关事物的干扰",
  "进入学习状态后先把手机移出视线，给注意力留下连续空间",
  "准备学习时先写下当前最重要的一件事，让开始动作更加清楚",
  "每次开始前不妨收好无关物品，避免刚进入状态就被打断",
  "状态有些分散时可以先调整坐姿和光线，帮助自己慢慢稳定下来",
  "想提高专注时可以关闭无关提醒，把精力留给眼前的事情",
  "坐下学习之前先准备好需要的资料，减少中途寻找物品",
  "准备重新开始时先做几次深呼吸，让自己更容易回到状态",
  "面对较长安排时可以把待办事项简单列好，避免同时处理太多内容",
  "计划下一轮学习时不妨选择安静的位置，为持续投入做好准备",
  "觉得注意力容易飘走时可以把容易打断自己的东西暂时收起",
  "进入下一阶段前可以检查设备和材料是否齐全，减少临时离开",
  "学习过程中感到杂乱时先把想法简单记下，避免一直在脑中切换",
  "想减少临时打断时可以把需要的东西放在手边，让后续安排更顺利",
  "开始当天的学习前不妨给自己设定清楚的开始信号，进入状态会更容易",
  "需要恢复节奏时可以先清理周围的干扰，给自己一个轻松的起点"
];

const ADVICE_CLAUSE_2 = [
  "注意力出现波动时可以把当前事情分成更小的步骤，不要因为一次波动否定自己",
  "遇到暂时不想继续的时刻不妨先完成眼前最容易的一项，帮助自己重新找到节奏",
  "感觉效率下降时可以用纸笔记下突然想到的事情，减少注意力分散",
  "出现想切换任务的念头时先把剩余安排重新排个顺序，让下一步更加明确",
  "需要重新集中注意力时可以短暂离开屏幕再回来，给当前状态一点恢复时间",
  "中途状态变化时可以放慢速度检查当前进展，避免因为着急而频繁切换",
  "完成一个阶段后可以给自己一个明确的结束点，让剩余安排更加从容",
  "不确定下一步做什么时先处理最重要的部分，把精力用在真正重要的事情上",
  "暂时无法保持专注时可以把已经完成的内容简单回顾，帮助自己看见进展",
  "想减少来回切换时不妨暂时搁置不紧急的事项，让注意力回到当前事情",
  "连续投入一段时间后可以用短暂休息恢复精力，比强行坚持更容易持续",
  "状态逐渐恢复时可以把下一步写得更具体，使后面的选择更加清楚",
  "注意到自己开始走神时先把困难部分做个标记，减少重新开始时的压力",
  "任务变得杂乱时可以减少同时处理的内容，让当前状态慢慢恢复",
  "重新回到学习时不妨从刚才中断的位置继续，帮助自己保持连续投入",
  "想让节奏慢慢稳定时可以为剩余安排选择合适的方式，不必要求一次做到完美"
];

const ADVICE_CLAUSE_3 = [
  "结束学习后可以记录一个做得好的地方，让进步能够被自己看见",
  "每天完成安排后不妨写下一个需要改进的地方，帮助坚持变得更自然",
  "准备休息时先保留今天有效的安排，为下一次继续积累信心",
  "回顾当天表现时可以给自己一句积极的评价，避免因为一次结果感到气馁",
  "下一次开始前可以把下一步计划简单记下，让调整有清楚的方向",
  "看到一点进步时不妨回想刚才最专注的时刻，把小变化慢慢积累起来",
  "状态不理想时也可以用平和的心态看待波动，有助于保持长期动力",
  "想长期坚持时先把困难拆成可以完成的小事，让下一次行动更容易开始",
  "完成一段学习后可以整理一下自己的收获，逐步建立适合自己的习惯",
  "给自己反馈时不妨认可已经完成的部分，保持对进步的耐心",
  "准备调整习惯时可以选择一个最容易执行的变化，让目标和状态更加匹配",
  "学习告一段落后先把值得继续的方法留下，帮助自己持续看到变化",
  "希望保持动力时可以把当天的体验简单记录，避免把短期波动看得过重",
  "发现方法有效时不妨回顾哪些做法帮助了自己，让每一次坚持都有意义",
  "需要重新规划时可以把还没完成的事情分阶段安排，为长期改善留下依据",
  "准备继续积累时可以给接下来的学习留出空间，让计划更容易落地"
];

const ENCOURAGEMENT_1 = [
  "你已经完成了今天的一段努力",
  "每一次愿意调整都是进步",
  "当前的表现已经说明你在认真坚持",
  "即使状态有起伏也没有关系",
  "能够发现问题本身就是很好的开始",
  "你已经为自己的目标迈出了一步",
  "一点一点保持下去就会看到变化",
  "愿意重新开始就值得肯定",
  "今天的记录可以成为下一次的参考",
  "稳定的积累比偶尔的冲刺更重要",
  "你对学习状态的关注本身很有价值",
  "每一次完成都在帮助你建立信心",
  "即使只改善一个小地方也很有意义",
  "你已经拥有继续进步的基础",
  "把注意力带回当前就是一种能力",
  "能够坚持记录说明你很重视成长"
];

const ENCOURAGEMENT_2 = [
  "继续保持这份耐心",
  "给自己多一些积极的肯定",
  "相信稳定的努力会带来回报",
  "用平和的心态迎接下一次",
  "从一个容易做到的改变开始",
  "把这份动力带到接下来的安排中",
  "允许自己按合适的节奏前进",
  "继续为自己的目标留出时间",
  "把今天的收获变成明天的起点",
  "用持续的小步调整代替急于求成",
  "保持对自己的信任",
  "在下一次学习中再试一种方法",
  "把已经做到的部分继续巩固",
  "给未来的自己留下一点信心",
  "继续关注那些真正有效的做法",
  "保持开始和完成的习惯"
];

const ENCOURAGEMENT_3 = [
  "你一定会越来越好",
  "相信自己能够做到",
  "接下来的表现值得期待",
  "你的坚持会留下清晰的答案",
  "一步一步来就好",
  "你会逐渐看见自己的变化",
  "请继续相信自己的能力",
  "你已经在向更好的状态靠近",
  "未来一定会有新的收获",
  "你的努力不会被浪费",
  "慢慢来同样能够走得很远",
  "愿你保持这份积极和耐心",
  "你会找到更适合自己的方法",
  "今天的坚持值得被肯定",
  "继续保持这份向上的力量",
  "相信每次尝试都有意义"
];

// 生成数据时使用不含逗号的短句，再统一用逗号连接，保持与人工样例一致。
const CONCISE_ADVICE_1 = [
  "学习开始前先整理好周围环境",
  "进入学习状态后先把手机移出视线",
  "准备学习时先写下当前最重要的一件事",
  "每次开始前先收好无关物品",
  "状态分散时先调整坐姿和光线",
  "想提高专注时先关闭无关提醒",
  "坐下学习之前先准备好需要的资料",
  "准备重新开始时先做几次深呼吸",
  "面对较长安排时先把待办事项列好",
  "计划下一轮学习时先选择安静的位置",
  "注意力容易飘走时先收起干扰物",
  "进入下一阶段前先检查设备和材料",
  "感到杂乱时先把想法简单记下",
  "想减少临时打断时先把需要的东西放在手边",
  "开始当天学习前先设定清楚的开始信号",
  "需要恢复节奏时先清理周围的干扰"
];

const CONCISE_ADVICE_2 = [
  "注意力出现波动时先完成眼前的一小步",
  "暂时不想继续时先处理最容易的一项",
  "感觉效率下降时先记下突然想到的事情",
  "想切换任务时先重新排好剩余安排",
  "需要重新集中时先短暂离开屏幕",
  "中途状态变化时先放慢速度检查进展",
  "完成一个阶段后给自己设定清楚的结束点",
  "不确定下一步时先处理最重要的部分",
  "暂时无法专注时先回顾已经完成的内容",
  "想减少切换时先搁置不紧急的事项",
  "连续投入后用短暂休息恢复精力",
  "状态逐渐恢复时把下一步写得更具体",
  "开始走神时先给困难部分做个标记",
  "任务杂乱时先减少同时处理的内容",
  "重新回来时先从刚才中断的位置继续",
  "想稳定节奏时先选择适合自己的方式"
];

const CONCISE_ADVICE_3 = [
  "结束学习后记录一个做得好的地方",
  "每天完成安排后写下一个需要改进的地方",
  "准备休息时保留今天有效的安排",
  "回顾当天表现时给自己一句积极的评价",
  "下一次开始前把计划简单记下",
  "看到进步时回想刚才最专注的时刻",
  "状态不理想时用平和的心态看待波动",
  "想长期坚持时把困难拆成小事",
  "完成一段学习后整理自己的收获",
  "给自己反馈时先认可已经完成的部分",
  "准备调整习惯时选择容易执行的变化",
  "学习告一段落后留下值得继续的方法",
  "希望保持动力时记录当天的体验",
  "发现方法有效时回顾帮助自己的做法",
  "需要重新规划时把未完成的事分阶段安排",
  "准备继续积累时给接下来的学习留出空间"
];

const CONCISE_ENCOURAGEMENT = [
  "相信自己会越来越好",
  "继续保持这份耐心",
  "你的坚持值得肯定",
  "一步一步来就好",
  "你会逐渐看见变化",
  "请继续相信自己的能力",
  "你正在靠近更好的状态",
  "未来一定会有新的收获",
  "你的努力不会被浪费",
  "慢慢来同样能够走得很远",
  "保持积极和耐心",
  "你会找到适合自己的方法",
  "今天的坚持值得认可",
  "继续保持向上的力量",
  "每次尝试都有意义",
  "你已经做得比想象中更好"
];

function fill(text, values) {
  return text.replace(/\{(\w+)\}/g, (_, key) => String(values[key]));
}

function makeAnalysis(info, rng, seen) {
  const efficiencyPool = info.base >= 90 ? EFFICIENCY_HIGH : info.base >= 75 ? EFFICIENCY_MIDDLE : EFFICIENCY_LOW;
  for (let attempt = 0; attempt < 1000; attempt++) {
    const efficiency = fill(pick(rng, efficiencyPool), {base: info.base});
    let distraction;
    if (info.eventTotal === 0) {
      distraction = pick(rng, DISTRACTION_NONE);
    } else {
      distraction = fill(pick(rng, DISTRACTION_PREFIX), {
        count: info.eventTotal,
        events: info.eventText
      });
      distraction += `并且${pick(rng, DISTRACTION_SUFFIX)}`;
    }
    const conclusion = pick(rng, CONCLUSION[info.grade] || CONCLUSION["一般"]);
    const result = `${efficiency}，${distraction}，${conclusion}，${pick(rng, ANALYSIS_DETAILS)}。`;
    if (!seen.has(result)) {
      seen.add(result);
      return result;
    }
  }
  throw new Error("AI分析句子组合不足");
}

function makeAdvice(rng, seen) {
  for (let attempt = 0; attempt < 1000; attempt++) {
    const result = [
      pick(rng, CONCISE_ADVICE_1),
      pick(rng, CONCISE_ADVICE_2),
      pick(rng, CONCISE_ADVICE_3),
      `${pick(rng, CONCISE_ENCOURAGEMENT)}！`
    ].join("，");
    if (!seen.has(result)) {
      seen.add(result);
      return result;
    }
  }
  throw new Error("AI建议句子组合不足");
}

function makeRow(sourceRow, analysis, advice) {
  const row = JSON.parse(JSON.stringify(sourceRow));
  const prefix = row[2].content.split("AI分析：")[0];
  row[2].content = `${prefix}AI分析：${analysis}AI建议：${advice}`;
  return JSON.stringify(row);
}

if (!fs.existsSync(SOURCE_PATH)) throw new Error(`找不到源数据：${SOURCE_PATH}`);
if (!fs.existsSync(SEED_PATH)) throw new Error(`找不到前20条样例：${SEED_PATH}`);

const sourceRows = fs.readFileSync(SOURCE_PATH, "utf8").trim().split(/\r?\n/).map(JSON.parse);
const seedRows = fs.readFileSync(SEED_PATH, "utf8").trim().split(/\r?\n/).map(JSON.parse);
if (sourceRows.length < SIZE) throw new Error(`源数据不足${SIZE}条`);
if (seedRows.length !== 20) throw new Error("前20条样例数量不正确");

const rng = makeRng(SEED);
const seenAnalysis = new Set();
const seenAdvice = new Set();
const output = [];

for (const row of seedRows) {
  const assistant = row[2].content;
  seenAnalysis.add(assistant.split("AI分析：")[1].split("AI建议：")[0]);
  seenAdvice.add(assistant.split("AI建议：")[1]);
  output.push(JSON.stringify(row));
}

for (let i = 20; i < SIZE; i++) {
  const row = sourceRows[i];
  const info = parseUser(row[1].content);
  const analysis = makeAnalysis(info, rng, seenAnalysis);
  const advice = makeAdvice(rng, seenAdvice);
  output.push(makeRow(row, analysis, advice));
}

fs.mkdirSync(path.dirname(OUTPUT_PATH), {recursive: true});
fs.writeFileSync(OUTPUT_PATH, `${output.join("\n")}\n`, "utf8");
console.log(`Generated ${output.length} records: ${OUTPUT_PATH}`);
console.log(`Unique full analysis strings: ${seenAnalysis.size}`);
console.log(`Unique full advice strings: ${seenAdvice.size}`);
