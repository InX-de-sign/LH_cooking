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
  'recentFrames', 'openRecipeDialog', 'submitRecipe', 'addTimer', 'tickTimers', 'stopAlarm', 'handleSwipe', 'setView',
  'sendFeedback', 'openMyRecipes', 'changeServings', 'withPrimer', 'toggleListen', 'decideRecipeSave'];

function harness(options = {}) {
  const elements = new Map();
  const images = [], revoked = [], draws = [];
  const sockets = [], windowEvents = {}, clearedIntervals = [], recognitions = [];
  const animationFrames = [], scrollCalls = [], resizeObservers = [];
  const desktopMedia = { matches: (options.width || 390) >= 700, listeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; } };
  const storage = new Map(Object.entries({ lh_cooking_client_id: 'test', lh_intro_seen: 'yes', ...(options.storage || {}) }));
  function element(tag) {
    const classes = new Set();
    const result = {
      textContent: '', innerHTML: '', dataset: {}, checked: false, children: [], attrs: {},
      value: '', hidden: false, listeners: {}, open: false, disabled: false,
      options: [], tagName: (tag || '').toUpperCase(),
      scrollHeight: 2000, clientHeight: 800, scrollTop: 0,
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
    requestAnimationFrame(fn) { animationFrames.push(fn); },
    scrollTo(position, y) {
      const root = context.document.documentElement;
      const top = typeof position === 'object' ? position.top : y;
      scrollCalls.push(top);
      root.scrollTop = Math.max(0, Math.min(top, root.scrollHeight - root.clientHeight));
    },
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; this.targets = []; resizeObservers.push(this); }
      observe(target) { assert.ok(target); this.targets.push(target); }
    },
    visualViewport: { height: 800, offsetTop: 0, listeners: {}, addEventListener(name, fn) { this.listeners[name] = fn; } },
    innerHeight: 800,
    matchMedia(query) { assert.equal(query, '(min-width: 700px)'); return desktopMedia; },
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
  context.removeEventListener = (name, callback) => { if (windowEvents[name] === callback) delete windowEvents[name]; };
  context.window = context;
  context.document.getElementById('jumpNewest').parentElement = element('footer');
  // Expose internals and skip opening a real socket in this harness.
  const instrumented = source.replace(/\n  connect\(\);\n\}\)\(\);/,
    `\n  window.test = { ${EXPORTS.join(', ')} };\n})();`);
  assert.notEqual(instrumented, source);
  vm.runInNewContext(instrumented, context);
  return { ...context.test, speech, timers, images, revoked, draws, sockets, windowEvents, clearedIntervals,
           recognitions, storage, body: context.document.body, root: context.document.documentElement,
           scrollCalls, resizeObservers, viewport: context.visualViewport,
           resizeTo(width) { desktopMedia.matches = width >= 700; desktopMedia.listeners.change?.(); },
           nextFrame() { const frame = animationFrames.splice(0); frame.forEach(fn => fn()); },
           flushFrames() { while (animationFrames.length) animationFrames.shift()(); } };
}
// A connected socket that records what the page sends.
function online(h) {
  const sent = [];
  h.state.ws = { readyState: 1, send(body) { sent.push(JSON.parse(body)); }, close() { this.closed = true; } };
  return sent;
}
const click = (el, target = {}) => el.listeners.click({ target, preventDefault() {} });
const button = (dataset) => ({ closest: () => ({ dataset }), dataset });
// Arrays made inside the page's vm context have another realm's prototype.
const plain = (value) => JSON.parse(JSON.stringify(value));

async function main() {
{
  const recipes = Array.from({ length: 13 }, (_, i) => ({ id: i + 1,
    name: `Soup ${i + 1}`, slug: `soup-${i + 1}`, category: 'Soup' }));
  const choose = { recipe: null, step: null, total_steps: 0 };
  for (const count of [0, 1, 6, 7, 13]) {
    const h = harness({ recipes: recipes.slice(0, count), width: 700 }); online(h);
    await h.renderCooking(choose);
    assert.equal(h.els.recipePicks.dataset.layout, 'grid');
    assert.equal(h.els.recipePicks.children.length, Math.min(count, 6));
    assert.equal(h.els.recipePicks.hidden, count === 0);
    assert.equal(h.els.seeMoreRecipes.hidden, count <= 6);
    assert.equal(h.els.recipeCount.textContent, `${count} ${count === 1 ? 'recipe' : 'recipes'}`);
    if (count > 6) {
      click(h.els.seeMoreRecipes);
      assert.equal(h.els.recipePicks.children.length, Math.min(count, 12));
      assert.equal(h.els.seeMoreRecipes.hidden, count <= 12);
      if (count > 12) {
        click(h.els.seeMoreRecipes);
        assert.equal(h.els.recipePicks.children.length, 13);
        assert.equal(h.els.seeMoreRecipes.hidden, true);
      }
    }
  }
  const h = harness({ recipes, width: 1200 }); const sent = online(h);
  await h.renderCooking(choose);
  click(h.els.seeMoreRecipes);
  const hearting = click(h.els.recipePicks.children[0].children[1]);
  h.handle({ type: 'settings_result', request_id: sent.at(-1).request_id, favorite: true }); await hearting;
  assert.equal(h.els.recipePicks.children.length, 12, 'favourite update preserves expansion');
  await h.renderCooking({ recipe: 'Soup 1', step: null, total_steps: 1 });
  await h.renderCooking(choose);
  assert.equal(h.els.recipePicks.children.length, 12, 'returning from details preserves expansion');
  h.resizeTo(699);
  assert.equal(h.els.recipePicks.dataset.layout, 'cards');
  assert.equal(h.els.recipePicks.children.length, 13, 'phone exposes every match');
  assert.equal(h.els.seeMoreRecipes.hidden, true);
  h.resizeTo(700);
  assert.equal(h.els.recipePicks.children.length, 12, 'desktop expansion survives resize');
  click(h.els.recipeCategories.children.find(b => b.textContent === 'Soup'));
  assert.equal(h.els.recipePicks.children.length, 6, 'category change resets expansion');
  click(h.els.seeMoreRecipes);
  h.els.recipeSearch.value = 'Soup'; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.children.length, 6, 'search resets expansion');
  h.resizeTo(390);
  assert.equal(h.els.recipePicks.children.length, 13);
  assert.equal(h.els.recipeSearch.value, 'Soup');
  assert.equal(h.els.recipeCategories.children.find(b => b.textContent === 'Soup').getAttribute('aria-pressed'), 'true');
  assert.equal(h.els.seeMoreRecipes.hidden, true);
}
{
  // Both ideas requests keep the discovery results at the top throughout replies.
  const h = harness(); const sent = online(h); h.settings.autoVoice = false;
  await h.renderCooking({ recipe: null, step: null, total_steps: 0 });
  h.flushFrames();
  for (const message of ['Give me 2 ideas for what to cook', 'Give me 2 different ideas']) {
    h.root.scrollTop = 900;
    click(h.els.ideasButton); h.flushFrames();
    assert.equal(sent.at(-1).message, message);
    assert.equal(Object.hasOwn(sent.at(-1), 'scrollTop'), false, 'scroll preference stays client-only');
    assert.equal(h.root.scrollTop, 0, 'request returns discovery to top');
    h.handle({ type: 'thinking' }); h.flushFrames();
    assert.equal(h.root.scrollTop, 0, 'loading stays at top');
    h.handle({ type: 'delta', text: 'Try these recipes.' }); h.flushFrames();
    assert.equal(h.root.scrollTop, 0, 'streaming stays at top');
    h.handle({ type: 'response', message: 'Try these recipes.',
      cooking: { recipe: null, step: null, total_steps: 0 },
      suggestions: [{ name: 'Egg Fried Rice' }, { name: 'Mapo Tofu' }] });
    await Promise.resolve(); h.flushFrames();
    assert.equal(h.els.ideaCards.hidden, false);
    assert.equal(h.els.ideasButton.textContent, '2 different ideas');
    assert.equal(h.root.scrollTop, 0, 'completed suggestions stay at top');
    h.root.scrollHeight += 200;
    h.resizeObservers[0].callback(); h.els.coach.listeners.load();
    h.viewport.listeners.resize(); h.flushFrames();
    assert.equal(h.root.scrollTop, 0, 'late layout and keyboard changes stay at top');
  }
  h.ask('Can I use tofu?'); h.flushFrames();
  assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight, 'ordinary prompts resume bottom scrolling');
  h.handle({ type: 'response', message: 'Yes.' }); h.flushFrames();
  assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight, 'ordinary replies follow newest');
}
{
  // Discovery filters combine, cards stay in one row, hearts do not start cooking.
  const recipes = [
    { id: 1, name: 'Garlic Noodles', slug: 'garlic-noodles', category: 'Main course', ingredients: ['garlic'], card_image: '/noodles.webp' },
    { id: 2, name: 'Tomato Soup', slug: 'tomato-soup', category: 'Soup', ingredients: ['tomatoes'] },
    { id: 3, name: 'Chicken Soup', slug: 'chicken-soup', category: 'Soup', ingredients: ['chicken'] },
  ];
  const h = harness({ recipes }); const sent = online(h);
  await h.renderCooking({ recipe: null, step: null, total_steps: 0 });
  assert.equal(h.els.recipePicks.children.length, 3);
  h.handle({ type: 'response', message: 'Welcome to your kitchen.' }); h.flushFrames();
  assert.equal(h.root.scrollTop, 0, 'the first greeting leaves discovery at the top');
  const firstImage = h.els.recipePicks.children[0].children[0].children[0].children[0];
  firstImage.listeners.error();
  assert.equal(firstImage.hidden, true, 'failed images reveal a designed placeholder');
  click(h.els.recipeCategories.children.find(b => b.textContent === 'Soup'));
  assert.equal(h.els.recipeCount.textContent, '2 recipes');
  h.els.recipeSearch.value = 'chicken'; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.children[0].dataset.recipe, 'Chicken Soup');
  assert.equal(h.els.recipeCount.textContent, '1 recipe');
  assert.equal(h.els.recipePicks.dataset.layout, 'cards');
  assert.equal(/id="(?:cardView|gridView)"/.test(html), false, 'discovery has no layout switch');
  assert.equal(h.storage.get('lh_recipe_category'), 'Soup');
  const before = JSON.stringify(h.state.cooking);
  const hearting = click(h.els.recipePicks.children[0].children[1]);
  const request = sent.at(-1);
  assert.deepEqual([request.type, request.source, request.slug], ['recipe_favorite', 'catalog', 'chicken-soup']);
  h.handle({ type: 'settings_result', request_id: request.request_id, favorite: true }); await hearting;
  assert.equal(JSON.stringify(h.state.cooking), before, 'favouriting does not change cooking state');
  assert.equal(sent.filter(r => r.type === 'ask' || r.type === 'recipe_open').length, 0);
  assert.equal(h.els.recipePicks.children[0].children[1].getAttribute('aria-pressed'), 'true');
  const restored = harness({ recipes, storage: { ...Object.fromEntries(h.storage), lh_recipe_layout: 'grid' } });
  await restored.renderCooking({ recipe: null, step: null, total_steps: 0 });
  assert.equal(restored.els.recipePicks.dataset.layout, 'cards', 'old grid preference cannot restore multiple rows');
  assert.equal(restored.els.recipePicks.children.length, 2);
  const favorites = h.openMyRecipes('favorites');
  h.handle({ type: 'settings_result', request_id: sent.at(-1).request_id, recipes: [
    { ...recipes[2], source: 'catalog', favorite: true },
    { id: 9, name: 'My soup', source: 'personal', favorite: true, category: 'Personal' },
  ] }); await favorites;
  assert.equal(h.els.myRecipesTitle.textContent, 'Favourite recipes');
  const cards = h.els.myRecipesList.children.map(g => g.children[1].children[0]);
  assert.equal(cards[0].children.length, 2, 'catalogue recipe has no personal star');
  assert.equal(cards[1].children[2].getAttribute('aria-label'), 'Personal recipe');
  assert.equal(cards[1].children[1].getAttribute('aria-pressed'), 'true', 'legacy personal favourites use hearts');
}
{
  const h = harness(); online(h); h.flushFrames();
  assert.equal(h.body.dataset.view, 'recipe');
  assert.ok(h.resizeObservers[0].targets.includes(h.els.stage), 'observe the visible cooking page, not only the hidden log');
  h.ask('Show me the latest prompt');
  h.nextFrame();
  assert.equal(h.root.scrollTop, 1200, 'sending jumps in recipe view too');
  h.root.scrollHeight = 2600;
  h.nextFrame();
  assert.equal(h.root.scrollTop, 1800, 'second frame uses height after composer/layout changes');
  assert.equal(h.body.dataset.view, 'recipe', 'sending preserves the selected view');
  h.root.scrollTop = 0;
  h.handle({ type: 'thinking' }); h.flushFrames();
  assert.equal(h.root.scrollTop, 1800, 'thinking is visible in recipe view');
  h.root.scrollTop = 0;
  h.handle({ type: 'response', message: 'Here is the newest reply.' }); h.flushFrames();
  assert.equal(h.root.scrollTop, 1800, 'completed reply jumps in recipe view');
  h.root.scrollTop = 0;
  h.handle({ type: 'response' }); h.flushFrames();
  assert.equal(h.root.scrollTop, 1800, 'completion without additional text still jumps');
  h.root.scrollHeight = 3000;
  h.resizeObservers[0].callback(); h.flushFrames();
  assert.equal(h.root.scrollTop, 2200, 'late cooking card layout keeps the reply visible');
  h.root.scrollHeight = 3400;
  h.els.coach.listeners.load(); h.flushFrames();
  assert.equal(h.root.scrollTop, 2600, 'camera frame loading in the coach line follows to bottom');
  h.root.scrollTop = 0;
  h.viewport.height = 500; h.viewport.listeners.resize(); h.flushFrames();
  assert.equal(h.root.scrollTop, 2600, 'keyboard resize follows the cooking session reply');
  h.setView('recipe'); h.flushFrames();
  h.resizeObservers[0].callback(); h.flushFrames();
  assert.equal(h.root.scrollTop, 0, 'explicitly opening the recipe clears cooking view follow');
  h.setView('chat'); h.nextFrame();
  h.root.scrollHeight = 3200; h.nextFrame();
  assert.equal(h.root.scrollTop, 2400, 'conversation opening settles at the final bottom');
}
{
  // The recipe coach panel and AI bubbles share the coach class. A broad
  // conversation-view hiding rule would hide the replies despite correct DOM.
  const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
  assert.doesNotMatch(css, /body\[data-view="chat"\]\s+\.coach\s*\{[^}]*display:\s*none/,
    'conversation view must not hide AI bubbles through their shared coach class');
  const h = harness(); online(h); h.setView('chat');
  h.ask('How do I cook eggs?');
  h.handle({ type: 'thinking' });
  h.handle({ type: 'delta', text: 'Beat the eggs.' });
  const streamed = h.els.log.children.find(el => el.className === 'msg coach');
  assert.ok(streamed);
  assert.equal(streamed.children[0].children[0], 'Beat the eggs.');
  h.handle({ type: 'response', message: 'Beat the eggs. Then fry them.' });
  assert.equal(h.els.log.children.filter(el => el.className === 'msg coach').length, 1);
  assert.equal(streamed.children[0].children[0], 'Beat the eggs. Then fry them.');
  h.setView('recipe'); h.setView('chat');
  assert.equal(h.els.log.children[0].textContent, 'How do I cook eggs?');
  assert.ok(h.els.log.children.includes(streamed), 'AI reply survives switching views');
  h.handle({ type: 'response', message: 'Serve warm.' });
  assert.equal(h.els.log.children.at(-1).children[0].children[0], 'Serve warm.');
}
{
  const h = harness(); online(h);
  h.flushFrames();
  h.scrollCalls.length = 0; // Ignore the initial recipe view's scroll to top.
  assert.equal(h.scrollCalls.length, 0, 'recipe view does not auto-scroll');
  h.setView('chat');
  assert.equal(h.scrollCalls.length, 0, 'wait for layout before scrolling');
  h.flushFrames();
  assert.equal(h.root.scrollTop, 1200);
  assert.equal(h.els.jumpNewest.hidden, true);
  const scrollUp = () => { h.root.scrollTop = 100; h.windowEvents.scroll(); };
  scrollUp();
  assert.equal(h.els.jumpNewest.hidden, false);
  click(h.els.jumpNewest);
  assert.equal(h.root.scrollTop, 1200, 'arrow jumps immediately');
  assert.equal(h.els.jumpNewest.hidden, true);
  scrollUp();
  assert.equal(h.ask('Help me cook'), true);
  h.flushFrames();
  assert.equal(h.root.scrollTop, 1200, 'sending overrides earlier scroll position');
  scrollUp(); h.handle({ type: 'thinking' }); h.flushFrames();
  assert.equal(h.root.scrollTop, 1200, 'typing indicator jumps to newest');
  scrollUp();
  const before = h.scrollCalls.length;
  h.handle({ type: 'delta', text: 'Heat ' });
  h.handle({ type: 'delta', text: 'the pan.' });
  h.root.scrollHeight = 2400;
  h.flushFrames();
  assert.equal(h.scrollCalls.length, before + 2, 'stream updates coalesce into one two-frame layout scroll');
  assert.equal(h.root.scrollTop, 1600);
  scrollUp(); h.handle({ type: 'response', message: 'Heat the pan. Then add oil.' }); h.flushFrames();
  assert.equal(h.root.scrollTop, 1600, 'final streamed response also jumps');
  scrollUp(); h.handle({ type: 'response', message: 'Ready.' }); h.flushFrames();
  assert.equal(h.root.scrollTop, 1600, 'non-streamed response jumps');
  scrollUp(); h.root.scrollHeight = 2800; h.els.log.listeners.load(); h.flushFrames();
  assert.equal(h.root.scrollTop, 2000, 'delayed image load jumps after new layout');
  scrollUp(); h.resizeObservers[0].callback(); h.flushFrames();
  assert.equal(h.root.scrollTop, 2000, 'log or dock resize keeps newest visible');
  scrollUp(); h.viewport.height = 500; h.viewport.listeners.resize(); h.flushFrames();
  assert.equal(h.root.style['--kb'], '300px');
  assert.equal(h.root.scrollTop, 2000, 'keyboard layout change jumps to newest');
  scrollUp(); h.els.textInput.scrollHeight = 120; h.els.textInput.listeners.input(); h.flushFrames();
  assert.equal(h.els.textInput.style.height, '123px');
  assert.equal(h.root.scrollTop, 2000, 'multiline composer resize jumps after layout');
  h.root.scrollTop = 1936; h.windowEvents.scroll();
  assert.equal(h.els.jumpNewest.hidden, true, '64px threshold is inclusive');
  h.root.scrollTop = 1935; h.windowEvents.scroll();
  assert.equal(h.els.jumpNewest.hidden, false);
  h.handle({ type: 'delta', text: 'Queued update' });
  h.setView('recipe'); h.flushFrames();
  assert.equal(h.root.scrollTop, 0, 'queued scroll does not affect recipe view');
  assert.equal(h.els.jumpNewest.hidden, true);
  h.els.log.listeners.load(); h.resizeObservers[0].callback(); h.flushFrames();
  assert.equal(h.root.scrollTop, 0, 'hidden conversation layout does not move recipe view');
}
{
  // Opening details wins over the accompanying reply, even from conversation.
  const h = harness(); online(h); h.settings.autoVoice = false;
  const recipe = { recipe: 'Noodles', step: null, total_steps: 2,
    steps: ['Boil noodles.', 'Serve.'], recipe_revision: 1 };
  h.setView('chat');
  h.ask("Let's make noodles"); h.flushFrames();
  h.handle({ type: 'response', message: 'Here is your recipe.', cooking: recipe });
  h.flushFrames();
  assert.equal(h.body.dataset.view, 'recipe', 'opening a recipe displays its details');
  assert.equal(h.root.scrollTop, 0, 'opening details stays at top after the reply');
  h.root.scrollHeight += 200;
  h.resizeObservers[0].callback(); h.els.coach.listeners.load();
  h.viewport.listeners.resize(); h.flushFrames();
  assert.equal(h.root.scrollTop, 0, 'late layout changes keep opened details at top');
  h.ask('Use rice noodles instead');
  h.handle({ type: 'response', message: 'Updated.', cooking: { ...recipe, recipe_revision: 2 } });
  h.flushFrames();
  assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight,
    'editing an open recipe still follows the answer to the bottom');
  h.ask("Let's make noodles", { openRecipe: true });
  h.handle({ type: 'response', message: 'Opened again.', cooking: { ...recipe, recipe_revision: 3 } });
  h.flushFrames();
  assert.equal(h.root.scrollTop, 0, 'reopening the same recipe also returns to top');
}
{
  // Start cooking, Next and Back keep the progress and large step visible throughout replies.
  const h = harness(); online(h); h.settings.autoVoice = false;
  const cooking = { recipe: 'Eggs', step: 0, total_steps: 3,
    steps: ['Beat.', 'Fry.', 'Serve.'], step_text: 'Beat.' };
  await h.renderCooking(cooking); h.flushFrames();
  for (const [index, trigger] of [() => click(h.els.startCooking), () => h.ask('Start cooking'),
    () => click(h.els.navNext), () => h.ask('next.'),
    () => h.ask('Next step please'), () => h.handleSwipe(-120, 0),
    () => click(h.els.navBack), () => h.ask('back.'), () => h.ask('Go back a step'),
    () => h.ask('Previous step please'), () => h.handleSwipe(120, 0)].entries()) {
    h.state.awaitingReply = false;
    if (index < 2) await h.renderCooking({ ...cooking, step: null });
    h.setView('chat'); h.flushFrames();
    trigger(); h.flushFrames();
    assert.equal(h.body.dataset.view, 'recipe', 'step navigation shows the cooking instructions');
    assert.equal(h.root.scrollTop, 0, 'step navigation immediately returns to the top');
    h.handle({ type: 'thinking' }); h.flushFrames();
    assert.equal(h.root.scrollTop, 0, 'thinking does not pull navigation away from the step');
    h.handle({ type: 'delta', text: 'Fry the eggs.' }); h.flushFrames();
    assert.equal(h.root.scrollTop, 0, 'streaming preserves the top');
    const step = index < 2 ? 0 : index < 6 ? 1 : 0;
    h.handle({ type: 'response', message: cooking.steps[step], cooking: { ...cooking, step, step_text: cooking.steps[step] } });
    h.flushFrames();
    assert.equal(h.els.stepText.textContent, cooking.steps[step]);
    assert.equal(h.els.progress.children[step].className, 'now');
    assert.equal(h.root.scrollTop, 0, 'final answer preserves the new step at top');
    h.root.scrollHeight += 200;
    h.resizeObservers[0].callback(); h.els.coach.listeners.load();
    h.viewport.listeners.resize(); h.flushFrames();
    assert.equal(h.root.scrollTop, 0, 'late layout and keyboard changes preserve navigation at top');
  }
  h.ask('What should I watch for next time?'); h.flushFrames();
  assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight, 'ordinary questions containing next still go to bottom');
  h.handle({ type: 'response', message: 'Watch the heat.' }); h.flushFrames();
  assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight, 'ordinary replies resume bottom scrolling');
  h.root.scrollTop = 0;
  h.ask('Should I put it back in the oven?'); h.flushFrames();
  assert.equal(h.root.scrollTop, h.root.scrollHeight - h.root.clientHeight, 'ordinary questions containing back still go to bottom');
}
{
  // Process names follow timers through navigation, reload, pause and alarms.
  const h = harness(); online(h); h.settings.autoVoice = false; h.els.voiceOut.checked = true;
  const cooking = { recipe: 'Spaghetti', step: 0, total_steps: 2, steps: ['Boil spaghetti.', 'Toss.'],
    step_text: 'Boil spaghetti.', timer_seconds: 600, timer_label: 'Boiling spaghetti' };
  await h.renderCooking(cooking);
  assert.equal(h.els.timerBtn.textContent, 'Start Boiling spaghetti timer (10:00)');
  click(h.els.timerBtn);
  const first = h.state.timers[0];
  assert.equal(first.label, 'Boiling spaghetti');
  assert.equal(h.els.timerBtn.disabled, true, 'existing step countdown prevents a duplicate');
  assert.match(h.els.timerBtn.textContent, /Boiling spaghetti timer:/);
  assert.match(h.els.timerStrip.children[0].children[0].textContent, /Boiling spaghetti/);
  await h.renderCooking({ ...cooking, step: 1, step_text: 'Toss.', timer_seconds: 120, timer_label: 'Tossing spaghetti' });
  click(h.els.timerBtn);
  assert.deepEqual(plain(h.state.timers.map(t => t.label)), ['Boiling spaghetti', 'Tossing spaghetti']);
  await h.renderCooking(cooking);
  assert.equal(h.els.timerBtn.disabled, true, 'returning to a step finds its original timer');
  assert.equal(first.label, 'Boiling spaghetti', 'navigation preserves the running timer name');
  click(h.els.timerStrip, { dataset: { timer: first.id } });
  assert.ok(first.pausedLeft > 0);
  assert.equal(first.label, 'Boiling spaghetti');
  click(h.els.timerStrip, { dataset: { timer: first.id } });
  assert.equal(first.pausedLeft, null);
  const reloaded = harness({ storage: { lh_cooking_timers: h.storage.get('lh_cooking_timers') } });
  assert.deepEqual(plain(reloaded.state.timers.map(t => t.label)), ['Boiling spaghetti', 'Tossing spaghetti']);
  first.endAt = Date.now() - 1000; h.tickTimers();
  assert.equal(h.els.alarmLabel.textContent, 'Boiling spaghetti');
  assert.ok(h.speech.said.includes("Time's up: Boiling spaghetti."));
  click(h.els.alarm);
  assert.deepEqual(plain(h.state.timers.map(t => t.label)), ['Tossing spaghetti']);
}
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
  assert.equal(h.els.chips.hidden, true, 'prep has its own Start button: no extra chips');
  assert.equal(h.els.prepFooter.hidden, false);
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
  const cooking = { recipe: 'Chicken Curry', step: 1, total_steps: 4, serves: 2, base_serves: 2, timer_seconds: 90, timer_label: 'Browning chicken',
    steps: ['Heat oil.', 'Brown chicken.', 'Add sauce.', 'Simmer.'], step_text: 'Brown chicken.', done_when: 'golden all over',
    ingredients: ['300 grams chicken'] };
  await h.renderCooking({ ...cooking, step: null });
  assert.equal(h.els.handsFree.checked, false);
  await h.renderCooking(cooking);
  assert.equal(h.body.dataset.stage, 'cook');
  assert.equal(h.body.dataset.view, 'recipe', 'cooking shows the recipe view');
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
  assert.equal(h.els.timerBtn.textContent, 'Start Browning chicken timer (1:30)');
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
  assert.equal(h.els.chips.hidden, true, 'choosing uses the buttons on the card, not chips');
  assert.equal(h.els.ideasButton.textContent, 'Give me 2 ideas');
  h.handle({ type: 'response', message: 'Two ideas.', cooking: { recipe: null, step: null, total_steps: 0 }, suggestions: [
    { name: 'Egg Fried Rice', why: 'quick', time_minutes: 20, difficulty: 'easy' }, { name: 'Mapo Tofu', why: '' }] });
  await Promise.resolve();
  assert.equal(h.els.ideaCards.hidden, false);
  assert.deepEqual(h.els.ideaCards.children.map(c => c.dataset.recipe), ['Egg Fried Rice', 'Mapo Tofu']);
  assert.equal(h.els.ideaCards.children[0].children[1].textContent, '20 min · easy');
  assert.equal(h.els.ideasButton.textContent, '2 different ideas', 'the big button now offers other ideas');
  click(h.els.ideasButton);
  assert.equal(sent.at(-1).message, 'Give me 2 different ideas');
  h.handle({ type: 'response', message: 'Or these.' });
  click(h.els.ideaCards, button({ recipe: 'Mapo Tofu' }));
  assert.equal(sent.at(-1).message, "Let's make Mapo Tofu");
  // The full recipe book is visible immediately, with combined search/filtering.
  assert.equal(h.els.recipePicks.children.length, 11, 'recipes are visible before searching');
  for (const term of ['curry', 'curry chicken', 'coconut']) {
    h.els.recipeSearch.value = term; h.els.recipeSearch.listeners.input();
    assert.equal(h.els.recipePicks.children.length, 1, `search matches ${term}`);
    assert.equal(h.els.recipePicks.children[0].dataset.recipe, 'Chicken Curry');
  }
  h.els.recipeSearch.value = 'noodle'; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.children.length, 9, 'all matching recipes are available');
  assert.equal(h.els.recipeCount.textContent, '9 recipes');
  h.els.recipeSearch.value = 'nothing matches'; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.hidden, true);
  assert.match(h.els.recipeSearchStatus.textContent, /No recipes found/);
  h.els.recipeSearch.value = ''; h.els.recipeSearch.listeners.input();
  assert.equal(h.els.recipePicks.children.length, 11, 'clearing search restores the recipe book');
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
  assert.equal(h.els.coachLine.children.at(-1).src, FRAME, 'and on the coach line');
  assert.equal(h.els.youSaid.textContent, 'Am I doing this right?', 'the cook sees what was heard');
  // The camera check: 3 photos over 3 seconds, sent as one burst.
  h.state.speechPending = 0;
  h.els.watch.checked = true;
  h.state.stage = 'cook';
  const before = sent.length;
  h.sendWatch();
  assert.equal(sent.length, before, 'nothing is sent until the burst is complete');
  assert.equal(h.els.watchStatus.textContent, 'Checking...');
  h.timers.at(-1)(); h.timers.at(-1)();
  assert.equal(sent.at(-1).type, 'watch');
  assert.equal(sent.at(-1).images.length, 3);
  const count = sent.length;
  h.state.awaitingReply = true; h.sendWatch();
  h.state.awaitingReply = false; h.state.stage = 'choose'; h.sendWatch();
  h.state.stage = 'done'; h.sendWatch();
  h.state.stage = 'cook'; h.els.watch.checked = false; h.sendWatch();
  assert.equal(sent.length, count, 'no checks while busy, choosing, finished, or when switched off');
  h.els.watch.checked = true; h.state.stage = 'prep';
  h.sendWatch(); h.state.awaitingReply = true; h.timers.at(-1)(); h.timers.at(-1)();
  assert.equal(sent.length, count, 'a burst is dropped if the cook starts talking meanwhile');
  h.state.awaitingReply = false;
  h.handle({ type: 'alert', level: 'hazard', message: 'Turn the heat down now.' });
  const alert = h.els.log.children.at(-1);
  assert.equal(alert.className, 'msg coach alert hazard');
  assert.equal(alert.children[0].textContent, 'Safety check');
  assert.equal(alert.children.at(-1).src, FRAME, 'alerts show the frame that triggered them');
  assert.equal(h.els.coachLine.className, 'coach-line alert hazard');
  assert.equal(h.els.coachLine.children[0].textContent, 'Safety check');
  assert.ok(h.speech.said.includes('Turn the heat down now.'), 'alerts are spoken when the cook is not busy');
}

{
  // A same-name revision replaces ingredients, steps and timer binding together.
  const h = harness();
  const original = { recipe: 'Noodles', recipe_revision: 1, step: 0, total_steps: 1,
    steps: ['Cook with soy sauce.'], step_text: 'Cook with soy sauce.',
    ingredients: ['200 grams noodles', '2 tablespoons soy sauce'], timer_seconds: 60, timer_label: 'Stir-frying noodles' };
  await h.renderCooking(original);
  h.state.checked.add(1); h.state.cameraTicked.add(1);
  click(h.els.timerBtn);
  assert.equal(h.els.timerBtn.disabled, true);
  const revised = { ...original, temporary: true, recipe_revision: 2,
    ingredients: ['200 grams noodles', '2 tablespoons tamari'],
    steps: ['Cook with tamari.'], step_text: 'Cook with tamari.', timer_seconds: 90, timer_label: 'Tossing noodles with tamari' };
  await h.renderCooking(revised);
  assert.equal(h.state.checked.size, 0, 'a replacement is not already gathered');
  assert.equal(h.state.cameraTicked.size, 0, 'old camera sightings do not apply to replacements');
  assert.equal(h.els.recipeName.textContent, 'Noodles (session recipe)');
  assert.equal(h.els.stepText.textContent, 'Cook with tamari.');
  assert.equal(h.els.timerBtn.textContent, 'Start Tossing noodles with tamari timer (1:30)');
  assert.equal(h.els.timerBtn.disabled, false, 'the old timer does not stand in for the revised step');
  assert.equal(h.state.timers.length, 1, 'existing timers remain available to the cook');
  assert.equal(h.state.timers[0].label, 'Stir-frying noodles', 'revisions do not rename running timers');
  h.state.checked.add(1);
  await h.renderCooking(revised);
  assert.equal(h.state.checked.size, 1, 'ordinary refreshes preserve the checklist');
}

{
  // Viewing My recipes is harmless, and the direct save button uses the same API.
  const h = harness(); const sent = online(h);
  assert.equal(h.els.unsavedRecipeStatus.hidden, true);
  const dirty = { recipe: 'Personal soup', temporary: true, saved_recipe_id: 9, recipe_revision: 2,
    draft_revision: 2, unsaved_changes: true, step: null, total_steps: 1, steps: ['Simmer.'],
    ingredients: ['1 cup stock'] };
  await h.renderCooking(dirty);
  assert.equal(h.els.unsavedRecipeStatus.hidden, false, 'unsaved adaptations are marked in the header');
  const viewing = h.openMyRecipes();
  const list = sent.at(-1);
  assert.equal(list.type, 'recipes_list');
  h.handle({ type: 'settings_result', request_id: list.request_id, recipes: [] });
  await viewing;
  assert.equal(h.state.cooking.unsaved_changes, true);
  assert.equal(h.els.recipeSaveDialog.open, false);
  click(h.els.closeMyRecipes);
  const saving = click(h.els.saveRecipeChanges);
  const request = sent.at(-1);
  assert.equal(request.type, 'recipe_save');
  assert.equal(h.els.saveRecipeChanges.disabled, true);
  h.handle({ type: 'settings_result', request_id: request.request_id, recipe_id: 9,
    cooking: { ...dirty, unsaved_changes: false } });
  await saving;
  assert.equal(h.els.saveRecipeChanges.hidden, true);
  assert.equal(h.els.unsavedRecipeStatus.hidden, true, 'saving clears the header reminder');
  assert.equal(h.state.cooking.recipe, dirty.recipe);
  assert.equal(h.els.recipeName.textContent, 'Personal soup (your recipe)');
  assert.equal(h.els.recipeSaveStatus.textContent, 'Saved to My recipes.');
  assert.equal(h.windowEvents.beforeunload, undefined);
}

{
  // Finish reminders appear once; the header back button guards unsaved changes.
  const h = harness(); const sent = online(h);
  const dirty = { recipe: 'Adapted noodles', temporary: true, recipe_revision: 3, draft_revision: 3,
    step: null, total_steps: 2, steps: ['Boil.', 'Toss.'], ingredients: ['200 grams noodles'], unsaved_changes: true };
  await h.renderCooking(dirty);
  assert.equal(h.els.saveRecipeChanges.hidden, false);
  assert.ok(h.windowEvents.beforeunload);
  const event = { preventDefault() { this.prevented = true; } };
  h.windowEvents.beforeunload(event);
  assert.equal(event.prevented, true);
  assert.equal(event.returnValue, '');
  await h.renderCooking({ ...dirty, step: 2 });
  assert.equal(h.els.recipeSaveDialog.open, true);
  assert.equal(h.els.recipeSaveConfirm.textContent, 'Save recipe');
  assert.equal(h.els.recipeSaveSkip.textContent, 'Not now');
  await h.decideRecipeSave('discard');
  await h.renderCooking({ ...dirty, step: 2 });
  assert.equal(h.els.recipeSaveDialog.open, false, 'the same finished version is reminded only once');
  click(h.els.backToRecipes);
  assert.equal(h.els.recipeSaveDialog.open, true);
  assert.equal(h.els.recipeSaveConfirm.textContent, 'Save and continue');
  assert.equal(h.els.recipeSaveCancel.hidden, false);
  await h.decideRecipeSave('cancel');
  assert.equal(h.state.cooking.recipe, dirty.recipe);
  assert.equal(h.els.backToRecipes.hidden, false);
  click(h.els.backToRecipes);
  const saving = h.decideRecipeSave('save');
  const request = sent.at(-1);
  assert.equal(request.type, 'recipe_save');
  assert.equal(request.expected_revision, 3);
  h.handle({ type: 'settings_result', request_id: request.request_id, recipe_id: 7,
    cooking: { ...dirty, step: 2, unsaved_changes: false, saved_recipe_id: 7 } });
  assert.equal(await saving, true);
  assert.equal(h.state.cooking.recipe, null, 'new chat continues after a successful save');
  assert.equal(h.body.dataset.stage, 'choose');
  assert.equal(h.els.backToRecipes.hidden, true);
  assert.equal(h.windowEvents.beforeunload, undefined);
}

{
  const h = harness(); online(h);
  assert.equal(h.els.backToRecipes.hidden, true, 'no back button on the recipe chooser');
  const recipe = { recipe: 'Soup', recipe_revision: 1, step: null, total_steps: 1,
    steps: ['Simmer.'], ingredients: ['Stock'] };
  await h.renderCooking(recipe);
  assert.equal(h.els.backToRecipes.hidden, false, 'back is available in preparation');
  await h.renderCooking({ ...recipe, step: 0 });
  assert.equal(h.els.backToRecipes.hidden, false, 'back is available while cooking');
  click(h.els.backToRecipes);
  assert.equal(h.body.dataset.stage, 'choose');
  assert.equal(h.els.recipeSaveDialog.open, false, 'unchanged recipes leave immediately');

  online(h);
  await h.renderCooking({ ...recipe, unsaved_changes: true });
  click(h.els.backToRecipes);
  const saving = h.decideRecipeSave('save');
  // A failed save must keep the user on the dish with the decision still open.
  const pendingId = [...h.state.requests.keys()][0];
  h.handle({ type: 'settings_result', request_id: pendingId, error: 'Save failed. Try again.' });
  assert.equal(await saving, false);
  assert.equal(h.state.cooking.recipe, 'Soup');
  assert.equal(h.els.recipeSaveDialog.open, true);
  assert.match(h.els.recipeSaveError.textContent, /Save failed/);
  await h.decideRecipeSave('discard');
  assert.equal(h.body.dataset.stage, 'choose');
  assert.equal(h.els.backToRecipes.hidden, true);
}

{
  // Failed saves keep the pending server navigation and the recipe intact.
  const h = harness(); const sent = online(h);
  const dirty = { recipe: 'My adapted soup', recipe_revision: 4, draft_revision: 4, temporary: true,
    saved_recipe_id: 8, unsaved_changes: true, step: null, total_steps: 1, steps: ['Simmer.'],
    ingredients: ['1 cup stock'], pending_leave: { token: 'leave-4', revision: 4 } };
  await h.renderCooking(dirty);
  assert.match(h.els.recipeSaveText.textContent, /replaces this recipe in My recipes/);
  const saving = h.decideRecipeSave('save');
  const first = sent.at(-1);
  assert.equal(first.type, 'recipe_leave');
  assert.equal(first.choice, 'save');
  assert.equal(first.token, 'leave-4');
  h.handle({ type: 'settings_result', request_id: first.request_id, error: 'Could not save. Try again.' });
  assert.equal(await saving, false);
  assert.equal(h.els.recipeSaveDialog.open, true);
  assert.match(h.els.recipeSaveError.textContent, /Try again/);
  assert.equal(h.state.cooking.recipe, dirty.recipe);
  assert.equal(h.state.savePrompt.pending.token, 'leave-4');
  const discard = h.decideRecipeSave('discard');
  const second = sent.at(-1);
  assert.equal(second.choice, 'discard');
  h.handle({ type: 'settings_result', request_id: second.request_id,
    cooking: { recipe: null, total_steps: 0, step: null, unsaved_changes: false, pending_leave: null } });
  assert.equal(await discard, true);
  assert.equal(h.els.recipeSaveDialog.open, false);
  assert.equal(h.windowEvents.beforeunload, undefined);
}

{
  // A later changed version is reminded again; viewing My recipes does not discard it.
  const h = harness(); online(h);
  const dirty = { recipe: 'Adapted soup', recipe_revision: 5, draft_revision: 5, unsaved_changes: true,
    step: 1, total_steps: 1, steps: ['Simmer.'], ingredients: ['1 cup stock'] };
  await h.renderCooking(dirty);
  await h.decideRecipeSave('discard');
  await h.renderCooking({ ...dirty, recipe_revision: 6, draft_revision: 6 });
  assert.equal(h.els.recipeSaveDialog.open, true, 'a different changed version gets a new reminder');
  await h.decideRecipeSave('cancel');
  h.newChat();
  await h.decideRecipeSave('discard');
  assert.equal(h.state.cooking.recipe, null, 'leaving without saving starts the requested new chat');
}

{
  // What the camera check saw shows quietly on the card; prep ticks ingredients it can see.
  const h = harness();
  const sent = online(h);
  await h.renderCooking({ recipe: 'Noodles', step: null, total_steps: 3, steps: ['Boil.', 'Fry.', 'Toss.'],
    ingredients: ['200 g noodles', '4 garlic cloves', '2 tbsp soy sauce'] });
  h.handle({ type: 'watch_status', status: 'on_track', activity: 'laying out ingredients', ingredients_seen: [1, 3, 9] });
  assert.equal(h.els.watchStatus.hidden, false);
  assert.equal(h.els.watchStatus.textContent, 'On track · laying out ingredients');
  assert.equal(h.els.watchStatus.dataset.level, 'on_track');
  assert.equal(h.els.ingredientsCount.textContent, '2 of 3 ready', 'seen ingredients are ticked, unknown numbers ignored');
  assert.equal(h.els.ingredientsList.children[0].children[0].children[2].textContent, 'seen');
  assert.match(h.els.toast.textContent, /Camera ticked: 200 g noodles, 2 tbsp soy sauce/);
  assert.equal(h.speech.said.length, 0, 'on-track checks are silent');
  h.handle({ type: 'watch_status', status: 'unclear', activity: '' });
  assert.equal(h.els.watchStatus.textContent, "Can't see clearly");
  h.handle({ type: 'watch_status', status: 'unclear', activity: '' });
  assert.match(h.els.watchStatus.textContent, /Point the camera at the pan/, 'twice unclear: say how to fix it');
  await h.renderCooking({ recipe: 'Noodles', step: 0, total_steps: 3, steps: ['Boil.', 'Fry.', 'Toss.'], step_text: 'Boil.',
    ingredients: ['200 g noodles', '4 garlic cloves', '2 tbsp soy sauce'] });
  h.handle({ type: 'alert', level: 'ahead', message: "Looks like you're on step 3. Jump there?", goto_step: 3 });
  assert.equal(h.els.coachActions.hidden, false);
  const go = h.els.coachActions.children[0];
  assert.equal(go.textContent, 'Go to step 3');
  h.state.awaitingReply = false;
  go.listeners.click();
  assert.equal(sent.at(-1).message, 'step 3');
  assert.equal(h.els.coachActions.hidden, true);
}

{
  // One view at a time: the recipe with the latest reply, or the whole conversation.
  const h = harness();
  online(h);
  h.state.audioUnlocked = true;
  h.ask('What can I use instead of butter?', { noFrames: true });
  h.handle({ type: 'thinking' });
  assert.equal(h.els.coachLine.children[1].textContent, 'Thinking...');
  h.handle({ type: 'delta', text: 'Use oil. ' });
  h.handle({ type: 'response', message: 'Use oil. Same amount.' });
  assert.equal(h.els.coachLine.hidden, false);
  assert.equal(h.els.coachLine.children[0].children[0], 'Use oil. Same amount.');
  assert.equal(h.body.dataset.view, 'recipe');
  click(h.els.chatButton);
  assert.equal(h.body.dataset.view, 'chat');
  assert.equal(h.els.chatView.hidden, false);
  assert.equal(h.els.chatButton.getAttribute('aria-pressed'), 'true');
  click(h.els.closeChat);
  assert.equal(h.body.dataset.view, 'recipe');
  assert.equal(h.els.textForm.hidden, true, 'voice first: typing is one tap away');
  click(h.els.keyboardToggle);
  assert.equal(h.els.textForm.hidden, false);
  assert.equal(h.els.keyboardToggle.getAttribute('aria-pressed'), 'true');
  // The typing box grows with the text; Enter sends, Shift+Enter makes a new line.
  h.els.textInput.scrollHeight = 120; h.els.textInput.value = 'Line one';
  h.els.textInput.listeners.input();
  assert.equal(h.els.textInput.style.height, '123px');
  let prevented = false;
  h.els.textInput.listeners.keydown({ key: 'Enter', shiftKey: true, preventDefault() { prevented = true; } });
  assert.equal(prevented, false, 'Shift+Enter keeps typing');
  h.handle({ type: 'response', message: 'ok' });
  h.els.textInput.listeners.keydown({ key: 'Enter', shiftKey: false, isComposing: false, preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(h.els.textInput.value, '', 'Enter sends and clears the box');
  assert.equal(h.els.youSaid.textContent, 'Line one');
  h.handle({ type: 'response', message: 'Done.' });
  h.ask('next'); h.ask('again');
  assert.equal(h.els.toast.hidden, false, 'notes pop up where they are seen');
  assert.match(h.els.toast.textContent, /wait for the current reply/);
  const noMic = harness({ noMic: true });
  assert.equal(noMic.els.textForm.hidden, false, 'without speech recognition the keyboard is shown');
  assert.match(noMic.els.hint.textContent, /type instead/);
}

{
  // First visit: a three-card intro, shown once.
  const h = harness({ storage: { lh_intro_seen: null } });
  assert.equal(h.els.introDialog.open, true);
  click(h.els.introNext); click(h.els.introNext);
  assert.equal(h.els.introNext.textContent, 'Get started');
  click(h.els.introNext);
  assert.equal(h.els.introDialog.open, false);
  assert.equal(h.storage.get('lh_intro_seen'), 'yes');
  assert.equal(harness().els.introDialog.open, false, 'not shown again');
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
  click(h.els.alarm);
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
  // My recipes: grouped cards, personal star badge, favourite heart, cook again.
  const h = harness();
  const sent = online(h);
  const opening = h.openMyRecipes();
  assert.equal(h.els.myRecipesDialog.open, true);
  h.handle({ type: 'settings_result', request_id: sent.at(-1).request_id, recipes: [
    { id: 7, name: '<b>Nan</b> noodles', favorite: false, steps: 5, serves: 2, time_minutes: 15 }] });
  await opening;
  const row = h.els.myRecipesList.children[0].children[1].children[0];
  const [cook, heart, badge] = row.children;
  assert.equal(cook.children[1].children[0].textContent, '<b>Nan</b> noodles', 'names are text, not HTML');
  assert.deepEqual(cook.children[1].children[1].children.map(c => c.textContent), ['15 mins', 'Serves 2']);
  assert.equal(badge.tagName, 'SPAN', 'the personal star is not an interactive toggle');
  assert.equal(badge.textContent, '\u2605');
  assert.equal(badge.getAttribute('aria-label'), 'Personal recipe');
  assert.equal(heart.getAttribute('aria-pressed'), 'false');
  const starring = click(heart);
  assert.deepEqual([sent.at(-1).type, sent.at(-1).recipe_id, sent.at(-1).favorite], ['recipe_favorite', 7, true]);
  h.handle({ type: 'settings_result', request_id: sent.at(-1).request_id, recipe_id: 7, favorite: true });
  await starring;
  assert.equal(h.els.myRecipesList.children[0].children[1].children[0].children[1].getAttribute('aria-pressed'), 'true');
  cook.listeners.click();
  assert.deepEqual(sent.at(-1), { type: 'recipe_open', recipe_id: 7 });
  assert.equal(h.els.myRecipesDialog.open, false);
  assert.equal(h.els.log.children.at(-1).textContent, 'Cook again: <b>Nan</b> noodles');
  h.setView('chat'); h.flushFrames();
  h.handle({ type: 'response', message: 'Ready to cook.', cooking: {
    recipe: '<b>Nan</b> noodles', saved_recipe_id: 7, recipe_revision: 1,
    step: null, total_steps: 1, steps: ['Cook noodles.'] } });
  h.flushFrames();
  assert.equal(h.body.dataset.view, 'recipe');
  assert.equal(h.root.scrollTop, 0, 'opening a saved recipe shows details at the top');
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
  click(h.els.camToggle);
  assert.equal(h.els.primerDialog.open, true);
  assert.match(h.els.primerText.textContent, /never saved/);
  click(h.els.primerCancel);
  assert.equal(h.storage.get('lh_primer_camera'), undefined, 'Not now asks again next time');
  click(h.els.camToggle);
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

console.log('PASS: browser behavior scenarios: category/search filters, horizontal recipe cards and legacy grid preference, favourites and personal badges, image fallback, bottom scrolling after layout in both views, conversation reply visibility, auto-scroll and jump-to-newest, stages, checklist, servings, cooking mode, swipes, voice defaults, idea cards, coach line, conversation view, keyboard toggle, intro, camera-check bursts and status, '
  + 'formatted replies, list speech, camera frames, watch alerts, timers and alarm, recipe upload, my recipes, settings, '
  + 'permission primers, offline banner, attachments, profile notes, history, new chats and reconnects.');
}

module.exports = main();
module.exports.catch((error) => { console.error(error); process.exit(1); });
