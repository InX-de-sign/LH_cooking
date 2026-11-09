// Offline behavior checks; run with `node tests/test_browser.cjs`.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../chatbot/static/index.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/\r\n/g, '\n');
let uuidSequence = 0;
const FRAME = 'data:image/jpeg;base64,YWJj';
// Elements the markup starts hidden, so the mock DOM starts the same way.
const HIDDEN_IDS = new Set([...html.matchAll(/<[a-z]+[^>]*\sid="([^"]+)"[^>]*>/g)]
  .filter((m) => /\shidden(\s|>|=)/.test(m[0])).map((m) => m[1]));
const EXPORTS = ['state', 'els', 'settings', 'handle', 'stopSpeaking', 'ask', 'attachPhoto', 'clearPhoto', 'openProfile',
  'saveProfile', 'loadHistory', 'loadHistoryDetail', 'connect', 'newChat', 'renderCooking', 'renderRich', 'sendWatch',
  'recentFrames', 'openRecipeDialog', 'submitRecipe', 'addTimer', 'tickTimers', 'stopAlarm', 'handleSwipe',
  'sendFeedback', 'openMyRecipes', 'changeServings', 'withPrimer', 'toggleListen'];

function harness(options = {}) {
  const elements = new Map();
  const images = [], revoked = [], draws = [];
  const sockets = [], windowEvents = {}, clearedIntervals = [], recognitions = [];
  const storage = new Map(Object.entries({ lh_cooking_client_id: 'test', ...(options.storage || {}) }));
  function element(tag) {
    const classes = new Set();
    const result = {
      textContent: '', innerHTML: '', dataset: {}, checked: false, children: [], attrs: {},
      value: '', hidden: false, listeners: {}, open: false, disabled: false,
      options: [], tagName: (tag || '').toUpperCase(),
      style: { setProperty(name, value) { this[name] = value; } },
      showModal() { this.open = true; }, close() { this.open = false; },
      classList: {
        add: (c) => classes.add(c), remove: (c) => classes.delete(c), contains: (c) => classes.has(c),
        toggle: (c, on) => ((on ?? !classes.has(c)) ? classes.add(c) : classes.delete(c)),
      },
      setAttribute(name, value) { this.attrs[name] = String(value); },
      getAttribute(name) { return this.attrs[name]; },
      addEventListener(name, fn) { this.listeners[name] = fn; },
      removeAttribute(name) { delete this[name]; }, click() { this.clicked = true; },
      appendChild(child) { this.children.push(child); child.parent = this; if (child.tagName === 'OPTION') this.options.push(child); },
      append(...children) { this.children.push(...children); },
      prepend(...children) { this.children.unshift(...children); },
      replaceChildren(...children) { this.children = children; },
      querySelectorAll() { return []; }, remove() {
        if (this.parent) {
          this.parent.children = this.parent.children.filter(child => child !== this);
          this.parent.options = this.parent.options.filter(child => child !== this);
        }
      },
    };
    if (tag === 'canvas') {
      result.getContext = () => ({ fillRect() {}, drawImage(...args) { draws.push(args); } });
      result.toDataURL = () => options.canvasData || FRAME;
    }
    return result;
  }
  const timers = [];
  const speech = {
    queue: [], said: [], utterances: [], getVoices() { return []; },
    cancel() { this.queue = []; },
    speak(utterance) { this.queue.push(utterance); this.said.push(utterance.text); this.utterances.push(utterance); },
  };
  class Recognition {
    constructor() { recognitions.push(this); }
    start() { this.started = true; this.onstart?.(); }
    stop() { this.onend?.(); }
    abort() { this.aborted = true; }
  }
  let intervalId = 100;
  const context = {
    document: {
      visibilityState: 'visible', addEventListener() {}, createElement: element,
      body: element('body'), documentElement: element('html'),
      getElementById(id) {
        if (!elements.has(id)) { const el = element(); el.hidden = HIDDEN_IDS.has(id); elements.set(id, el); }
        return elements.get(id);
      },
    },
    speechSynthesis: speech,
    SpeechSynthesisUtterance: function(text) { this.text = text; },
    webkitSpeechRecognition: options.noMic ? undefined : Recognition,
    localStorage: { getItem: (k) => (storage.has(k) ? storage.get(k) : null), setItem: (k, v) => storage.set(k, String(v)) },
    crypto: { randomUUID() { return `test-${++uuidSequence}`; } }, navigator: {},
    location: { protocol: 'http:', host: 'localhost:8000', search: options.search || '' },
    history: { replaceState() {} }, URLSearchParams,
    fetch: async () => ({ json: async () => ({ recipes: options.recipes || [] }) }),
    addEventListener(name, fn) { windowEvents[name] = fn; },
    URL: class extends URL {
      static createObjectURL() { return 'blob:test-photo'; }
      static revokeObjectURL(url) { revoked.push(url); }
    },
    Image: class {
      constructor() { this.naturalWidth = 4000; this.naturalHeight = 3000; images.push(this); }
      set src(value) { if (!options.delayImage) Promise.resolve().then(() => this.onload()); }
    },
    WebSocket: class {
      static OPEN = 1;
      constructor(url) { this.url = url; this.readyState = 0; this.sent = []; sockets.push(this); }
      send(body) { this.sent.push(JSON.parse(body)); }
      close() { this.closed = true; this.onclose?.(); }
    },
    setTimeout(fn) { timers.push(fn); }, clearTimeout() {},
    setInterval() { return ++intervalId; }, clearInterval(id) { clearedIntervals.push(id); },
  };
  context.window = context;
  // Expose internals and skip opening a real socket in this harness.
  const instrumented = source.replace(/\n  connect\(\);\n\}\)\(\);/,
    `\n  window.test = { ${EXPORTS.join(', ')} };\n})();`);
  assert.notEqual(instrumented, source);
  vm.runInNewContext(instrumented, context);
  return { ...context.test, speech, timers, images, revoked, draws, sockets, windowEvents, clearedIntervals,
           recognitions, storage, body: context.document.body, root: context.document.documentElement };
}
// A connected socket that records what the page sends.
function online(h) {
  const sent = [];
  h.state.ws = { readyState: 1, send(body) { sent.push(JSON.parse(body)); } };
  return sent;
}
const click = (el, target = {}) => el.listeners.click({ target, preventDefault() {} });
const button = (dataset) => ({ closest: () => ({ dataset }), dataset });
// Arrays made inside the page's vm context have another realm's prototype.
const plain = (value) => JSON.parse(JSON.stringify(value));

async function main() {
{
  // Prep: ingredients as a checklist, servings, and a clear Start.
  const h = harness();
  const sent = online(h);
  const preview = { recipe: 'Chicken Curry', step: null, total_steps: 4, serves: 2, base_serves: 2, time_minutes: 35,
    steps: ['Heat oil.', 'Brown chicken.', 'Add sauce.', 'Simmer.'],
    ingredients: ['300 grams chicken', '200 millilitres coconut milk', '<b>literal ingredient</b>'] };
  await h.renderCooking(preview);
  assert.equal(h.body.dataset.stage, 'prep');
  assert.equal(h.els.prepPanel.hidden, false);
  assert.equal(h.els.cookPanel.hidden, true);
  assert.equal(h.els.ingredientsPanel.open, true);
  assert.deepEqual(h.els.ingredientsList.children.map(li => li.dataset.ingredient), preview.ingredients);
  assert.equal(h.els.ingredientsList.children[2].innerHTML, '', 'ingredients use safe text rendering');
  assert.equal(h.els.ingredientsCount.textContent, '0 of 3 ready');
  h.els.ingredientsList.listeners.change({ target: { dataset: { index: '1' }, checked: true } });
  assert.equal(h.els.ingredientsCount.textContent, '1 of 3 ready');
  click(h.els.ingredientsList, { dataset: { missing: '200 millilitres coconut milk' } });
  assert.equal(sent.at(-1).message, "I don't have 200 millilitres coconut milk. What can I use instead?");
  h.handle({ type: 'response', message: 'Use cream.' });
  assert.equal(h.els.recipeDetails.textContent, 'Serves 2 · About 35 min');
  assert.equal(h.els.servingsRow.hidden, false);
  assert.equal(h.els.servingsValue.textContent, '2');
  assert.deepEqual(h.els.chips.children.map(b => b.textContent), ['Start cooking', 'Make it for 4', 'Something else']);
  assert.deepEqual(h.els.stepsList.children.map(li => li.textContent), preview.steps, 'every step is listed');
  const scaling = h.changeServings(1);
  assert.deepEqual([sent.at(-1).type, sent.at(-1).servings], ['set_servings', 3]);
  h.handle({ type: 'settings_result', request_id: sent.at(-1).request_id,
    cooking: { ...preview, serves: 3, ingredients: ['450 grams chicken', '300 millilitres coconut milk', '<b>literal ingredient</b>'] } });
  await scaling;
  assert.equal(h.els.servingsValue.textContent, '3');
  assert.equal(h.els.ingredientsCount.textContent, '1 of 3 ready', 'ticks survive rescaling');
  h.state.stream = {}; h.state.visionAvailable = true; h.els.video.videoWidth = 640; h.els.video.videoHeight = 480;
  click(h.els.startCooking);
  assert.equal(sent.at(-1).message, 'Start cooking');
  assert.equal(sent.at(-1).images, undefined, 'step commands do not spend data on camera frames');
}

{
  // Cook: cooking mode switches voice on, swipes and big buttons drive steps.
  const h = harness({ storage: { lh_primer_mic: 'yes' } });
  const sent = online(h);
  h.state.audioUnlocked = true;
  const cooking = { recipe: 'Chicken Curry', step: 1, total_steps: 4, serves: 2, base_serves: 2, timer_seconds: 90,
    steps: ['Heat oil.', 'Brown chicken.', 'Add sauce.', 'Simmer.'], step_text: 'Brown chicken.', done_when: 'golden all over',
    ingredients: ['300 grams chicken'] };
  await h.renderCooking({ ...cooking, step: null });
  assert.equal(h.els.handsFree.checked, false);
  await h.renderCooking(cooking);
  assert.equal(h.body.dataset.stage, 'cook');
  assert.equal(h.body.dataset.chat, 'collapsed', 'the chat shrinks to the latest reply');
  assert.equal(h.els.cookPanel.hidden, false);
  assert.equal(h.els.stepText.textContent, 'Brown chicken.');
  assert.equal(h.els.stepCount.textContent, 'Step 2 of 4');
  assert.equal(h.els.ingredientsPanel.open, false, 'ingredients collapse when cooking starts');
  assert.deepEqual(h.els.stepsList.children.map(li => li.className), ['done', 'now', '', '']);
  assert.equal(h.els.voiceOut.checked, true);
  assert.equal(h.els.handsFree.checked, true, 'hands-free turns on in cooking mode once the mic was allowed');
  assert.match(h.els.log.children.at(-1).textContent, /Cooking mode/);
  assert.ok(h.els.chips.children.some(b => b.dataset.say === 'Am I doing this right?'), 'camera check is one tap away');
  assert.equal(h.els.timerBtn.hidden, false);
  assert.equal(h.els.timerBtn.textContent, 'Start 1:30 timer');
  click(h.els.navNext);
  assert.equal(sent.at(-1).message, 'Next');
  h.handle({ type: 'response', message: 'Step 3.' });
  assert.equal(h.handleSwipe(20, 0), false, 'a small drag is not a swipe');
  assert.equal(h.handleSwipe(-120, 200), false, 'scrolling is not a swipe');
  assert.equal(h.handleSwipe(-120, 10), true);
  assert.equal(sent.at(-1).message, 'Next', 'swipe left = next');
  h.handle({ type: 'response', message: 'Step 3.' });
  h.handleSwipe(140, 5);
  assert.equal(sent.at(-1).message, 'Go back a step', 'swipe right = back');
  h.handle({ type: 'response', message: 'Step 2.' });
  h.state.swipedAt = 0;
  h.els.stepText.listeners.click();
  assert.equal(h.recognitions.at(-1).started, true, 'tapping the step starts listening');
  assert.equal(h.recognitions.at(-1).lang, 'en-US');
  h.recognitions.at(-1).stop();
  // Done: ask how it went, keep a note for next time, put voice settings back.
  await h.renderCooking({ ...cooking, step: 4 });
  assert.equal(h.body.dataset.stage, 'done');
  assert.equal(h.els.donePanel.hidden, false);
  assert.equal(h.els.handsFree.checked, false, 'leaving cooking mode restores the saved setting');
  click(h.els.ratingRow, button({ rating: 'loved' }));
  h.els.doneNote.value = '  less salt  ';
  assert.equal(h.sendFeedback(), true);
  assert.deepEqual(sent.at(-1), { type: 'cook_feedback', rating: 'loved', note: 'less salt' });
  assert.equal(h.els.feedbackDone.hidden, false);
}

{
  // Cooking mode without mic permission yet: speak, but don't start listening on its own.
  const h = harness();
  await h.renderCooking({ recipe: 'Eggs', step: 0, total_steps: 2, steps: ['Beat.', 'Cook.'], step_text: 'Beat.' });
  assert.equal(h.els.handsFree.checked, false);
  assert.match(h.els.log.children.at(-1).textContent, /Tap the mic/);
  h.toggleListen();
  assert.equal(h.els.primerDialog.open, true, 'explain the microphone before the browser asks');
  assert.equal(h.recognitions.length, 0);
  click(h.els.primerContinue);
  assert.equal(h.recognitions.length, 1);
  assert.equal(h.storage.get('lh_primer_mic'), 'yes');
}

{
  // Choose: idea cards come from the coach's suggest_recipes tool.
  const recipes = Array.from({ length: 9 }, (_, i) => ({ name: `Noodle Dish ${i}`, aliases: [], ingredients: ['noodles'] }));
  const h = harness({ recipes: [
    { name: 'Chicken Curry', aliases: ['curry chicken'], ingredients: ['coconut milk'], time_minutes: 35 },
    { name: 'Spaghetti Bolognese', aliases: ['spag bol'], ingredients: ['minced beef'], time_minutes: 45 },
    ...recipes,
  ] });
  const sent = online(h);
  await h.renderCooking({ recipe: null, step: null, total_steps: 0 });
  assert.equal(h.body.dataset.stage, 'choose');
  assert.equal(h.els.choosePanel.hidden, false);
  assert.equal(h.els.ideaCards.hidden, true);
  assert.ok(h.els.chips.children.some(b => b.dataset.action === 'upload'));
  assert.ok(h.els.chips.children.some(b => b.dataset.action === 'saved'));
  h.handle({ type: 'response', message: 'Two ideas.', cooking: { recipe: null, step: null, total_steps: 0 }, suggestions: [
    { name: 'Egg Fried Rice', why: 'quick', time_minutes: 20, difficulty: 'easy' }, { name: 'Mapo Tofu', why: '' }] });
  await Promise.resolve();
  assert.equal(h.els.ideaCards.hidden, false);
  assert.deepEqual(h.els.ideaCards.children.map(c => c.dataset.recipe), ['Egg Fried Rice', 'Mapo Tofu']);
  assert.equal(h.els.ideaCards.children[0].children[1].textContent, '20 min · easy');
  click(h.els.ideaCards, button({ recipe: 'Mapo Tofu' }));
  assert.equal(sent.at(-1).message, "Let's make Mapo Tofu");
  // The recipe book stays hidden until searched, and shows at most six.
  assert.equal(h.els.recipePicks.children.length, 0, 'the recipe book is not dumped on screen');
  for (const term of ['curry', 'curry chicken', 'coconut']) {
    h.els.recipeSearch.value = term; h.els.recipeSearch.listeners.input();
    assert.equal(h.els.recipePicks.children.length, 1, `search matches ${term}`);
    assert.equal(h.els.recipePicks.children[0].dataset.recipe, 'Chicken Curry');
  }
  h.els.recipeSearch.value = 'noodle'; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.children.length, 6, 'search results are capped');
  assert.match(h.els.recipeSearchStatus.textContent, /Top 6 of 9/);
  h.els.recipeSearch.value = 'nothing matches'; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.hidden, true);
  assert.match(h.els.recipeSearchStatus.textContent, /No recipes found/);
  h.els.recipeSearch.value = ''; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.hidden, true, 'clearing search hides the picks again');
}

{
  // Coach replies render lists as real lists, safely.
  const h = harness();
  const bubble = { children: [], replaceChildren(...c) { this.children = c; } };
  h.renderRich(bubble, 'Great choice.\n\n- 4 eggs\n- **1 tbsp** butter\n\n1. Beat the eggs.\n2. Melt butter.\n<img src=x onerror=alert(1)>');
  assert.deepEqual(bubble.children.map(c => c.tagName), ['P', 'UL', 'OL', 'P']);
  assert.equal(bubble.children[1].children.length, 2);
  assert.equal(bubble.children[1].children[1].children[0].tagName, 'STRONG');
  assert.equal(bubble.children[1].children[1].children[0].textContent, '1 tbsp');
  assert.equal(bubble.children[2].children[1].children[0], 'Melt butter.');
  assert.equal(bubble.children[3].children[0], '<img src=x onerror=alert(1)>', 'HTML in a reply stays text');
  assert.equal(bubble.children[3].innerHTML, '', 'nothing is written as HTML');
}

{
  // Numbered steps are spoken without "one dot"; ingredient lists stay on screen.
  const h = harness();
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'Here we go.\n- 4 eggs\n- butter\n- salt\n1. Beat the' });
  h.handle({ type: 'delta', text: ' eggs. Then rest.\n2. Melt **butter**.\n' });
  h.handle({ type: 'response', message: 'Here we go.\n- 4 eggs\n- butter\n- salt\n1. Beat the eggs. Then rest.\n2. Melt **butter**.\n' });
  assert.deepEqual(h.speech.said, ['Here we go.', 'The list is on your screen.', '1, Beat the eggs. Then rest.', '2, Melt butter.']);
  const h2 = harness();
  h2.handle({ type: 'response', message: 'Two ideas:\n- Egg fried rice: quick.\n- Mapo tofu: spicy.' });
  assert.deepEqual(h2.speech.said, ['Two ideas:', 'Egg fried rice: quick.', 'Mapo tofu: spicy.'], 'short bullet lists are read');
}

{
  const h = harness({ storage: { lh_primer_mic: 'yes' } });
  h.els.handsFree.checked = true;
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'Use 2.' });
  assert.equal(h.speech.said.length, 0, 'partial decimal is not spoken');
  h.handle({ type: 'delta', text: '5 cups. Then stir' });
  assert.deepEqual(h.speech.said, ['Use 2.5 cups.'], 'first sentence speaks before final response');
  h.speech.queue.shift().onend();
  assert.equal(h.timers.length, 0, 'hands-free must wait for stream completion');
  h.handle({ type: 'response', message: 'Use 2.5 cups. Then stir' });
  assert.deepEqual(h.speech.said, ['Use 2.5 cups.', 'Then stir'], 'final response is not repeated');
  assert.equal(h.els.log.children.filter(x => x.className === 'msg coach').length, 1,
    'streamed and final events share a single reply bubble');
  h.speech.queue.shift().onend();
  assert.equal(h.timers.length, 1, 'hands-free resumes only after speech finishes');
}

{
  const h = harness();
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'First. ' });
  h.stopSpeaking();
  h.handle({ type: 'delta', text: 'Second. ' });
  h.handle({ type: 'response', message: 'First. Second.' });
  assert.deepEqual(h.speech.said, ['First.'], 'stopping speech mutes subsequent chunks of that reply');
  assert.equal(h.state.awaitingReply, false);
}

{
  const h = harness();
  h.els.voiceOut.checked = false;
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'Quiet. ' });
  h.handle({ type: 'response', message: 'Quiet.' });
  assert.equal(h.speech.said.length, 0, 'voice-off mode stays silent');
}

{
  const h = harness();
  h.handle({ type: 'thinking' });
  const messages = h.els.log.children.length;
  h.ask('next');
  assert.equal(h.els.log.children.length, messages + 1);
  assert.equal(h.els.log.children.at(-1).className, 'msg note', 'a second request waits for the first');
}

{
  const h = harness();
  h.handle({ type: 'response', message: 'Cook tofu.', sources: [
    { url: 'https://recipes.example/tofu', title: '<script>recipe</script>' },
    { url: 'javascript:alert(1)', title: 'Unsafe' },
  ] });
  const bubble = h.els.log.children.find(x => x.className === 'msg coach');
  const sources = bubble.children.at(-1);
  const links = sources.children.filter(x => x && x.href);
  assert.equal(links.length, 1, 'unsafe source links are discarded');
  assert.equal(links[0].textContent, '<script>recipe</script>', 'source titles are rendered as text');
  assert.equal(links[0].rel, 'noopener noreferrer');
  assert.deepEqual(h.speech.said, ['Cook tofu.'], 'source URLs are not spoken');
}

{
  // Questions carry the last few seconds of camera; the answer shows what was seen.
  const h = harness();
  const sent = online(h);
  h.state.stream = {}; h.state.visionAvailable = true;
  h.els.video.videoWidth = 1280; h.els.video.videoHeight = 720;
  h.state.frames = ['data:image/jpeg;base64,b2xk', 'data:image/jpeg;base64,bWlk'];
  h.state.audioUnlocked = true;
  h.ask('Am I doing this right?');
  assert.deepEqual(sent[0].images, ['data:image/jpeg;base64,b2xk', 'data:image/jpeg;base64,bWlk', FRAME], 'two buffered frames plus one now, oldest first');
  assert.equal(sent[0].image, undefined);
  assert.equal(sent[0].language, 'en');
  h.handle({ type: 'thinking' });
  assert.equal(h.els.log.children.at(-1).children[0].textContent, 'Looking at your pan...', 'say what the coach is doing');
  h.handle({ type: 'response', message: 'Looks good.', vision: { summary: null, frames: 3, hazards: [] } });
  const reply = h.els.log.children.at(-1);
  assert.equal(reply.children.at(-2).textContent, 'Checked your camera');
  assert.equal(reply.children.at(-1).src, FRAME, 'the frame the coach judged is shown with its answer');
  h.state.speechPending = 0;
  h.els.watch.checked = true;
  h.state.cooking = { recipe: 'Eggs', step: 0, total_steps: 3 };
  h.sendWatch();
  assert.equal(sent.at(-1).type, 'watch');
  assert.equal(sent.at(-1).images.length, 3);
  const count = sent.length;
  h.state.awaitingReply = true; h.sendWatch();
  h.state.awaitingReply = false; h.state.cooking = { recipe: 'Eggs', step: null, total_steps: 3 }; h.sendWatch();
  h.state.cooking = { recipe: 'Eggs', step: 0, total_steps: 3 }; h.els.watch.checked = false; h.sendWatch();
  assert.equal(sent.length, count, 'no checks while busy, before starting, or when switched off');
  h.handle({ type: 'alert', level: 'hazard', message: 'Turn the heat down now.' });
  const alert = h.els.log.children.at(-1);
  assert.equal(alert.className, 'msg coach alert hazard');
  assert.equal(alert.children[0].textContent, 'Safety check');
  assert.equal(alert.children.at(-1).src, FRAME, 'alerts show the frame that triggered them');
  assert.ok(h.speech.said.includes('Turn the heat down now.'), 'alerts are spoken when the cook is not busy');
}

{
  // Timers: named, several at once, loud when done, survive a reload.
  const h = harness();
  h.handle({ type: 'response', message: 'Timer set.', timers: [{ label: 'Noodles', seconds: 180 }, { label: 'Eggs', seconds: 420 }] });
  assert.equal(h.els.timerStrip.hidden, false);
  assert.deepEqual(h.els.timerStrip.children.map(c => c.children[0].textContent), ['Noodles 3:00', 'Eggs 7:00']);
  const saved = JSON.parse(h.storage.get('lh_cooking_timers'));
  assert.deepEqual(saved.map(t => t.label), ['Noodles', 'Eggs']);
  const eggs = h.state.timers[1];
  click(h.els.timerStrip, { dataset: { timer: eggs.id, action: 'toggle' } });
  assert.ok(eggs.pausedLeft > 400, 'tapping a timer pauses it');
  assert.equal(h.els.timerStrip.children[1].className, 'timer-chip paused');
  h.state.timers[0].endAt = Date.now() - 1;
  h.tickTimers();
  assert.equal(h.els.alarm.hidden, false, 'a finished timer takes over the screen');
  assert.equal(h.els.alarmLabel.textContent, 'Noodles');
  assert.ok(h.speech.said.includes("Time's up: Noodles."));
  click(h.els.alarmDismiss);
  assert.equal(h.els.alarm.hidden, true);
  assert.deepEqual(plain(h.state.timers.map(t => t.label)), ['Eggs'], 'stopping the alarm clears that timer');
  click(h.els.timerStrip, { dataset: { timer: eggs.id, action: 'remove' } });
  assert.equal(h.els.timerStrip.hidden, true);
  const reloaded = harness({ storage: { lh_cooking_timers: JSON.stringify([
    { id: 'a', label: 'Rice', endAt: Date.now() - 5000, total: 600 }, { id: 'b', label: 'Stock', endAt: Date.now() + 60000, total: 900 }]) } });
  assert.deepEqual(plain(reloaded.state.timers.map(t => [t.label, Boolean(t.done)])), [['Rice', true], ['Stock', false]]);
  assert.equal(reloaded.els.alarm.hidden, true, 'a timer that ran out while closed is shown, not rung');
  assert.equal(reloaded.els.timerStrip.children[0].className, 'timer-chip done');
}

{
  // Upload: text, link or photo; the dialog says what's wrong before sending.
  const h = harness();
  const sent = online(h);
  h.openRecipeDialog();
  assert.equal(h.els.recipeDialog.open, true);
  h.submitRecipe({ preventDefault() {} });
  assert.equal(sent.length, 0);
  assert.equal(h.els.recipeStatus.dataset.error, 'true', 'empty upload is refused with a message');
  h.els.recipeUrl.value = 'not a link';
  h.submitRecipe({ preventDefault() {} });
  assert.match(h.els.recipeStatus.textContent, /https/);
  h.els.recipeUrl.value = 'https://recipes.example/noodles';
  h.submitRecipe({ preventDefault() {} });
  assert.deepEqual(sent[0], { type: 'recipe_upload', text: '', url: 'https://recipes.example/noodles' });
  assert.equal(h.els.recipeDialog.open, false);
  assert.equal(h.els.recipeUrl.value, '');
  h.handle({ type: 'thinking' });
  assert.equal(h.els.log.children.at(-1).children[0].textContent, 'Reading your recipe...');
  h.handle({ type: 'response', message: 'Got it.' });
  h.openRecipeDialog();
  h.els.recipeText.value = "Grandma's noodles\nBoil noodles 3 minutes.";
  h.submitRecipe({ preventDefault() {} });
  assert.deepEqual(sent[1], { type: 'recipe_upload', text: "Grandma's noodles\nBoil noodles 3 minutes." });
  assert.equal(h.state.awaitingReply, true);
}

{
  // My recipes: list, star, cook again.
  const h = harness();
  const sent = online(h);
  const opening = h.openMyRecipes();
  assert.equal(h.els.myRecipesDialog.open, true);
  h.handle({ type: 'settings_result', request_id: sent.at(-1).request_id, recipes: [
    { id: 7, name: '<b>Nan</b> noodles', favorite: false, steps: 5, serves: 2, time_minutes: 15 }] });
  await opening;
  const row = h.els.myRecipesList.children[0];
  const [star, info, cook] = row.children;
  assert.equal(info.children[0].textContent, '<b>Nan</b> noodles', 'names are text, not HTML');
  assert.equal(info.children[1].textContent, '5 steps · serves 2 · 15 min');
  assert.equal(star.getAttribute('aria-pressed'), 'false');
  const starring = star.listeners.click();
  assert.deepEqual([sent.at(-1).type, sent.at(-1).recipe_id, sent.at(-1).favorite], ['recipe_favorite', 7, true]);
  h.handle({ type: 'settings_result', request_id: sent.at(-1).request_id, recipe_id: 7, favorite: true });
  await starring;
  assert.equal(star.getAttribute('aria-pressed'), 'true');
  cook.listeners.click();
  assert.deepEqual(sent.at(-1), { type: 'recipe_open', recipe_id: 7 });
  assert.equal(h.els.myRecipesDialog.open, false);
  assert.equal(h.els.log.children.at(-1).textContent, 'Cook again: <b>Nan</b> noodles');
}

{
  // Settings: language, speech speed, text size and theme stick, and reach the coach.
  const h = harness();
  const sent = online(h);
  h.els.languageSelect.value = 'yue'; h.els.languageSelect.listeners.change();
  assert.deepEqual(sent.at(-1), { type: 'set_language', language: 'yue' });
  assert.equal(h.root.lang, 'zh-HK');
  h.els.rateSelect.value = '1.15'; h.els.rateSelect.listeners.change();
  click(h.els.textSize, { dataset: { scale: '1.3' } });
  assert.equal(h.root.style['--scale'], '1.3');
  h.els.themeSelect.value = 'contrast'; h.els.themeSelect.listeners.change();
  assert.equal(h.root.dataset.theme, 'contrast');
  h.els.themeSelect.value = 'auto'; h.els.themeSelect.listeners.change();
  assert.equal(h.root.dataset.theme, undefined);
  const stored = JSON.parse(h.storage.get('lh_cooking_settings'));
  assert.deepEqual([stored.language, stored.rate, stored.scale], ['yue', 1.15, 1.3]);
  h.state.audioUnlocked = true;
  h.handle({ type: 'response', message: '- 一\n- 二\n- 三' });
  const said = h.speech.utterances.at(-1);
  assert.deepEqual([said.text, said.lang, said.rate], ['清單喺螢幕上。', 'zh-HK', 1.15]);
  const reloaded = harness({ storage: { lh_cooking_settings: h.storage.get('lh_cooking_settings') } });
  reloaded.connect();
  assert.equal(new URL(reloaded.sockets[0].url).searchParams.get('lang'), 'yue', 'the greeting uses the saved language');
  const broken = harness({ storage: { lh_cooking_settings: '{"language":"xx","rate":9,"scale":"huge","theme":"neon"}' } });
  assert.deepEqual([broken.settings.language, broken.settings.rate, broken.settings.scale, broken.settings.theme], ['en', 1, 1, 'auto']);
}

{
  // The camera is explained before the browser's permission prompt.
  const h = harness();
  click(h.els.camOn);
  assert.equal(h.els.primerDialog.open, true);
  assert.match(h.els.primerText.textContent, /never saved/);
  click(h.els.primerCancel);
  assert.equal(h.storage.get('lh_primer_camera'), undefined, 'Not now asks again next time');
  click(h.els.camOn);
  click(h.els.primerContinue);
  assert.equal(h.storage.get('lh_primer_camera'), 'yes');
  assert.match(h.els.log.children.at(-1).textContent, /can't open the camera/, 'then the camera is requested');
}

{
  // A dropped connection is obvious, and can be retried at once.
  const h = harness(); h.connect();
  const socket = h.sockets[0];
  socket.readyState = 1; socket.onopen();
  assert.equal(h.els.offlineBanner.hidden, true);
  socket.readyState = 3; socket.onclose();
  assert.equal(h.els.offlineBanner.hidden, false);
  assert.match(h.els.offlineText.textContent, /Reconnecting in 1 s/);
  click(h.els.retryNow);
  assert.equal(h.sockets.length, 2, 'Retry now reconnects immediately');
  h.sockets[1].readyState = 1; h.sockets[1].onopen();
  assert.equal(h.els.offlineBanner.hidden, true);
}

{
  const h = harness();
  h.state.visionAvailable = true;
  h.els.uploadPhoto.listeners.click();
  assert.equal(h.els.photoInput.clicked, true, 'paperclip opens the photo picker');
  await h.attachPhoto({ name: 'dinner.jpg', type: 'image/jpeg', size: 5000 });
  assert.equal(h.state.queue.length, 0, 'selecting a photo does not send it');
  assert.equal(h.els.attachment.hidden, false);
  assert.equal(h.els.photoPreview.hidden, false);
  assert.equal(h.draws[0][3], 1280, 'large photo is resized before attachment');
  assert.equal(h.draws[0][4], 960);
  assert.equal(h.revoked.length, 1, 'temporary object URL is released');
  const photo = h.state.photo;
  await h.attachPhoto(null);
  assert.equal(h.state.photo, photo, 'cancelling the picker preserves the attachment');
  h.state.audioUnlocked = true;
  assert.equal(h.ask('What ingredients are here?'), true);
  const sent = JSON.parse(h.state.queue[0]);
  assert.equal(sent.image, photo.data);
  assert.equal(sent.image_source, 'upload');
  assert.equal(sent.message, 'What ingredients are here?');
  assert.equal(h.state.photo, null, 'photo clears after sending');
  assert.equal(h.els.attachment.hidden, true);
  h.state.awaitingReply = false;
  h.ask('next');
  assert.equal(JSON.parse(h.state.queue[1]).image, undefined, 'photo is not resent with the next turn');
}

{
  const h = harness();
  h.state.visionAvailable = true;
  await h.attachPhoto({ name: 'dinner.png', type: 'image/png', size: 5000 });
  h.state.awaitingReply = true;
  h.els.textInput.value = 'Keep my draft';
  h.els.textForm.listeners.submit({ preventDefault() {} });
  assert.equal(h.els.textInput.value, 'Keep my draft', 'busy submission preserves text');
  assert.ok(h.state.photo, 'busy submission preserves the photo');
  h.state.awaitingReply = false;
  h.state.visionAvailable = false;
  assert.equal(h.ask('Analyze'), false, 'no silent send without a vision model');
  assert.ok(h.state.photo);
  h.state.visionAvailable = true;
  h.state.audioUnlocked = true;
  h.ask('');
  assert.equal(JSON.parse(h.state.queue[0]).message, 'What can you see in this photo?', 'photo-only send has a useful prompt');
}

{
  const h = harness({ delayImage: true });
  const pending = h.attachPhoto({ name: 'dinner.webp', type: 'image/webp', size: 5000 });
  assert.equal(h.ask('Wait for photo'), false, 'sending waits for image preparation');
  h.els.removePhoto.listeners.click();
  h.images[0].onload();
  await pending;
  assert.equal(h.state.photo, null, 'removing a photo cancels an in-flight decode');
  assert.equal(h.els.attachment.hidden, true);
  assert.equal(h.revoked.length, 1);
  await h.attachPhoto({ name: 'movie.gif', type: 'image/gif', size: 10 });
  assert.equal(h.state.photo, null);
  assert.ok(h.els.log.children.at(-1).textContent.includes('JPEG, PNG or WebP'));
  await h.attachPhoto({ name: 'huge.jpg', type: 'image/jpeg', size: 20_000_001 });
  assert.ok(h.els.log.children.at(-1).textContent.includes('20 MB'));
}

{
  const h = harness();
  const requests = online(h);
  const opening = h.openProfile();
  assert.equal(h.els.profileDialog.open, true);
  assert.equal(h.els.saveProfile.disabled, true, 'cannot save before the profile loads');
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id,
    profile: { cook_name: 'Sam', skill_level: 'I can boil an egg', dietary: 'Peanut allergy', notes: 'Pasta: less salt' } });
  await opening;
  assert.equal(h.els.profileName.value, 'Sam');
  assert.equal(h.els.profileSkill.value, 'I can boil an egg', 'custom experience is retained');
  assert.equal(h.els.profileDietary.value, 'Peanut allergy');
  assert.equal(h.els.profileNotes.value, 'Pasta: less salt');
  h.els.profileName.value = 'Alex';
  const saving = h.saveProfile({ preventDefault() {} });
  assert.equal(requests.at(-1).profile.cook_name, 'Alex');
  assert.equal(requests.at(-1).profile.notes, 'Pasta: less salt', 'notes are saved with the profile');
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id, profile: { cook_name: 'Alex' } });
  await saving;
  assert.equal(h.els.profileDialog.open, false);
  assert.equal(h.state.requests.size, 0);
  assert.equal(h.speech.said.length, 0, 'settings are not spoken as chat replies');
}

{
  const h = harness();
  const requests = online(h);
  h.els.textInput.value = 'My unfinished question';
  const loading = h.loadHistory();
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id, next_offset: 20,
    sessions: [{ session_id: 'first', started_at: 1000, recipe_name: '<script>pasta</script>', turn_count: 2 }] });
  await loading;
  assert.equal(h.els.historyList.children[0].children[0].textContent, '<script>pasta</script>', 'history titles use text, not HTML');
  assert.equal(h.els.moreHistory.hidden, false);
  const detail = h.loadHistoryDetail('first');
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id, next_offset: 40,
    session: { recipe_name: 'Pasta' }, turns: [{ ts: 1000, user_text: 'Question', assistant_text: '<img src=x>', vision: 'a tomato' }] });
  await detail;
  assert.equal(h.els.historyDetail.hidden, false);
  assert.equal(h.els.historyTurns.children[0].children[2].children[1], '<img src=x>', 'history messages are plain text');
  assert.equal(h.els.olderHistory.hidden, false);
  const older = h.loadHistoryDetail('first', 40);
  assert.equal(requests.at(-1).offset, 40);
  h.handle({ type: 'settings_result', request_id: requests.at(-1).request_id, next_offset: null,
    session: { recipe_name: 'Pasta' }, turns: [{ ts: 900, user_text: 'Older question', assistant_text: 'Older reply' }] });
  await older;
  assert.equal(h.els.historyTurns.children[0].children[1].children[1], 'Older question');
  assert.equal(h.els.textInput.value, 'My unfinished question', 'history does not alter the current draft');
  assert.equal(h.els.log.children.length, 0, 'past turns do not enter the current conversation');
  assert.equal(h.speech.said.length, 0);
}

{
  const h = harness();
  await h.openProfile();
  assert.equal(h.els.profileStatus.dataset.error, 'true', 'offline profile loading has a clear error');
  assert.equal(h.els.saveProfile.disabled, true, 'offline loading cannot overwrite the saved profile with empty fields');
  await h.loadHistory();
  assert.equal(h.els.historyStatus.dataset.error, 'true');
}

{
  const h = harness();
  h.connect();
  const first = h.sockets[0];
  const firstChat = new URL(first.url).searchParams.get('chat_id');
  first.readyState = 1; first.onopen();
  h.state.visionAvailable = true;
  await h.attachPhoto({ name: 'old.jpg', type: 'image/jpeg', size: 5000 });
  h.els.textInput.value = 'Old dish draft';
  h.addTimer('Oven', 600);
  h.state.queue.push('old queued message');
  h.state.awaitingReply = true;
  h.handle({ type: 'delta', text: 'Old dish reply' });
  h.newChat();
  const next = h.sockets[1];
  assert.equal(first.closed, true);
  assert.notEqual(new URL(next.url).searchParams.get('chat_id'), firstChat);
  assert.equal(new URL(next.url).searchParams.get('client_id'), 'test', 'profile identity survives new chats');
  assert.equal(h.els.log.children.length, 0);
  assert.equal(h.els.textInput.value, '');
  assert.equal(h.state.photo, null);
  assert.equal(h.state.queue.length, 0);
  assert.equal(h.state.awaitingReply, false);
  assert.deepEqual(plain(h.state.timers.map(t => t.label)), ['Oven'], 'kitchen timers keep running across a new chat');
  assert.equal(h.body.dataset.stage, 'choose');
  assert.equal(h.timers.length, 0, 'closing the old socket does not schedule a reconnect');
  first.onmessage({ data: JSON.stringify({ type: 'response', message: 'Late old reply' }) });
  assert.equal(h.els.log.children.length, 0, 'old replies cannot enter the new dish chat');
  const anotherPage = harness(); anotherPage.connect();
  assert.notEqual(new URL(anotherPage.sockets[0].url).searchParams.get('chat_id'), firstChat, 'reopening gets a fresh chat');
}

{
  const h = harness(); h.connect();
  const socket = h.sockets[0];
  socket.onclose();
  h.timers.at(-1)();
  assert.equal(h.sockets[1].url, socket.url, 'temporary disconnect reconnects to the same chat');
  const current = h.sockets[1];
  h.windowEvents.pagehide();
  assert.equal(current.closed, true);
  assert.equal(h.state.exiting, true);
  h.windowEvents.pageshow({ persisted: true });
  assert.equal(h.state.exiting, false);
  assert.notEqual(h.sockets.at(-1).url, current.url, 'restoring a closed page from browser cache starts fresh');
}

console.log('PASS: 25 browser behavior scenarios: stages, checklist, servings, cooking mode, swipes, voice defaults, idea cards, '
  + 'formatted replies, list speech, camera frames, watch alerts, timers and alarm, recipe upload, my recipes, settings, '
  + 'permission primers, offline banner, attachments, profile notes, history, new chats and reconnects.');
}

main().catch((error) => { console.error(error); process.exit(1); });
